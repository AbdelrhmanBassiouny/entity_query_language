from __future__ import annotations

import codecs
import itertools
import os
import re
from collections import defaultdict, UserDict
from copy import copy
from dataclasses import dataclass, field
from subprocess import check_call
from tempfile import NamedTemporaryFile
from typing import Iterable, FrozenSet, Tuple, Dict, DefaultDict, Hashable, Generic, Optional, Any


try:
    import six
except ImportError:
    six = None

try:
    from graphviz import Source
except ImportError:
    Source = None

from anytree import Node, RenderTree, PreOrderIter
from typing_extensions import Callable, Set, Any, Optional, List, TypeVar

from . import logger


class IDGenerator:
    """
    A class that generates incrementing, unique IDs and caches them for every object this is called on.
    """

    _counter = 0
    """
    The counter of the unique IDs.
    """

    # @lru_cache(maxsize=None)
    def __call__(self, obj: Any) -> int:
        """
        Creates a unique ID and caches it for every object this is called on.

        :param obj: The object to generate a unique ID for, must be hashable.
        :return: The unique ID.
        """
        self._counter += 1
        return self._counter


@dataclass
class SeenSet:
    seen: List[Any] = field(default_factory=list, init=False)

    def add(self, assignment):
        """
        Add an assignment (dict of key→value).
        Missing keys are implicitly wildcards.
        Example: {"k1": "v1"} means all k2,... are allowed
        """
        self.seen.append(assignment)

    def check(self, assignment):
        """
        Check if an assignment (dict) is covered by seen entries.
        """
        for constraint in self.seen:
            if all(assignment[k] == v if k in assignment else False for k, v in constraint.items()):
                return True
        return False


T = TypeVar("T")

@dataclass
class HashedValue(Generic[T]):
    value: T
    id_: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.id_ is None:
            if hasattr(self.value, "_id_"):
                self.id_ = self.value._id_
            else:
                self.id_ = id(self.value)

    def __hash__(self):
        return hash(self.id_)

    def __eq__(self, other):
        return self.id_ == other.id_


@dataclass
class HashedIterable(Generic[T]):
    """
    A wrapper for an iterable that hashes its items.
    This is useful for ensuring that the items in the iterable are unique and can be used as keys in a dictionary.
    """
    iterable: Iterable[HashedValue[T]] = field(default_factory=list)
    values: Dict[int, HashedValue[T]] = field(default_factory=dict)

    def __post_init__(self):
        if self.iterable and not isinstance(self.iterable, HashedIterable):
            self.iterable = (HashedValue(id_=k, value=v) if not isinstance(v, HashedValue) else v
                             for k, v in enumerate(self.iterable))

    def get(self, key: int, default: Any) -> HashedValue[T]:
        return self.values.get(key, default)

    def add(self, value: Any):
        if not isinstance(value, HashedValue):
            value = HashedValue(value)
        if value.id_ not in self.values:
            self.values[value.id_] = value
        return self

    def update(self, iterable: Iterable[Any]):
        for v in iterable:
            self.add(v)

    def __iter__(self):
        """
        Iterate over the hashed values.

        :return: An iterator over the hashed values.
        """
        yield from self.values.values()
        for v in self.iterable:
            self.values[v.id_] = v
            yield v

    def __or__(self, other) -> HashedIterable[T]:
        return self.union(other)

    def __and__(self, other) -> HashedIterable[T]:
        return self.intersection(other)

    def intersection(self, other):
        common_keys = self.values.keys() & other.values.keys()
        common_values = {k: self.values[k] for k in common_keys}
        return HashedIterable(values=common_values)

    def difference(self, other):
        common_keys = self.values.keys() & other.values.keys()
        left_keys = self.values.keys() - other.values.keys()
        values = {k: self.values[k] for k in left_keys}
        return HashedIterable(values=values)

    def union(self, other):
        if not isinstance(other, HashedIterable):
            other = HashedIterable(values={HashedValue(v).id_: HashedValue(v) for v in make_list(other)})
        all_keys = self.values.keys() | other.values.keys()
        all_values = {k: self.values.get(k, other.values.get(k)) for k in all_keys}
        return HashedIterable(values=all_values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, id_: int) -> HashedValue:
        """
        Get the HashedValue by its id.

        :param id_: The id of the HashedValue to get.
        :return: The HashedValue with the given id.
        """
        return self.values[id_]

    def __setitem__(self, id_: int, value: HashedValue[T]):
        """
        Set the HashedValue by its id.

        :param id_: The id of the HashedValue to set.
        :param value: The HashedValue to set.
        """
        self.values[id_] = value

    def __copy__(self):
        """
        Create a shallow copy of the HashedIterable.

        :return: A new HashedIterable instance with the same values.
        """
        return HashedIterable(values=self.values.copy())

    def __contains__(self, item):
        return item in self.values

    def __hash__(self):
        return hash(tuple(sorted(self.values.keys())))

    def __eq__(self, other):
        keys_are_equal = self.values.keys() == other.values.keys()
        if not keys_are_equal:
            return False
        values_are_equal = all(my_v == other_v for my_v, other_v in zip(self.values.values(), other.values.values()))
        return values_are_equal

    def __bool__(self):
        return bool(self.values) or bool(self.iterable)


ALL = HashedValue(object())

class CacheDict(UserDict):
    ...

@dataclass
class IndexedCache:
    keys: List[int] = field(default_factory=list)
    seen_set: SeenSet = field(default_factory=SeenSet, init=False)
    cache: CacheDict = field(default_factory=CacheDict, init=False)
    enter_count: int = field(default=0, init=False)
    search_count: int = field(default=0, init=False)
    found_count: int = field(default=0, init=False)

    def __post_init__(self):
        self.keys.sort()

    def insert(self, assignment: Dict, output: Any):
        assignment = dict(assignment)
        cache = self.cache
        for k_idx, k in enumerate(self.keys):
            if k not in assignment.keys():
                raise ValueError(f"Missing key {k} in assignment {assignment}")
            if k_idx < len(self.keys) - 1:
                if (k, assignment[k]) not in cache:
                    cache[(k, assignment[k])] = CacheDict()
                cache = cache[(k, assignment[k])]
            else:
                cache[(k, assignment[k])] = output
        self.seen_set.add(assignment)

    def check(self, assignment: Dict) -> bool:
        """
        Check if seen entries cover an assignment (dict).

        :param assignment: The assignment to check.
        """
        return self.seen_set.check(assignment)

    def retrieve(self, assignment, cache=None, key_idx=0, result: Dict = None) -> Iterable:
        result = result or copy(assignment)
        if cache is None:
            cache = self.cache
        self.enter_count += 1
        key = self.keys[key_idx]
        while key in assignment:
            cache = cache[(key, assignment[key])]
            if key_idx+1 < len(self.keys):
                key_idx = key_idx + 1
                key = self.keys[key_idx]
            else:
                break
        if key not in assignment:
            for cache_key, cache_val in cache.items():
                result = copy(result)
                result[key] = cache_key[1]
                if isinstance(cache_val, CacheDict):
                    yield from self.retrieve(assignment, cache_val, key_idx + 1, result)
                else:
                    yield result, cache_val
        else:
            yield result, cache


def filter_data(data, selected_indices):
    data = iter(data)
    prev = -1
    encountered_indices = set()
    for idx in selected_indices:
        if idx in encountered_indices:
            continue
        encountered_indices.add(idx)
        skip = idx - prev - 1
        data = itertools.islice(data, skip, None)
        try:
            yield next(data)
        except StopIteration:
            break
        prev = idx


def make_list(value: Any) -> List:
    """
    Make a list from a value.

    :param value: The value to make a list from.
    """
    return list(value) if is_iterable(value) else [value]


def is_iterable(obj: Any) -> bool:
    """
    Check if an object is iterable.

    :param obj: The object to check.
    """
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, type, bytes, bytearray))


def make_tuple(value: Any) -> Any:
    """
    Make a tuple from a value.
    """
    return tuple(value) if is_iterable(value) else (value,)


def make_set(value: Any) -> Set:
    """
    Make a set from a value.

    :param value: The value to make a set from.
    """
    return set(value) if is_iterable(value) else {value}


def get_unique_node_names_func(root_node) -> Callable[[Node], str]:
    nodes = [root_node]

    def get_all_nodes(node):
        for c in node.children:
            nodes.append(c)
            get_all_nodes(c)

    get_all_nodes(root_node)

    def nodenamefunc(node: Node):
        """
        Set the node name for the dot exporter.
        """
        similar_nodes = [n for n in nodes if n.name == node.name]
        node_idx = similar_nodes.index(node)
        return node.name if node_idx == 0 else f"{node.name}_{node_idx}"

    return nodenamefunc


def edge_attr_setter(parent, child):
    """
    Set the edge attributes for the dot exporter.
    """
    if child and hasattr(child, "weight") and child.weight is not None:
        return f'style="bold", label=" {child.weight}"'
    return ""


_RE_ESC = re.compile(r'["\\]')


class FilteredDotExporter(object):

    def __init__(self, node, include_nodes=None, graph="digraph", name="tree", options=None,
                 indent=4, nodenamefunc=None, nodeattrfunc=None,
                 edgeattrfunc=None, edgetypefunc=None, maxlevel=None):
        """
        Dot Language Exporter.

        Args:
            node (Node): start node.

        Keyword Args:
            graph: DOT graph type.

            name: DOT graph name.

            options: list of options added to the graph.

            indent (int): number of spaces for indent.

            nodenamefunc: Function to extract node name from `node` object.
                          The function shall accept one `node` object as
                          argument and return the name of it.

            nodeattrfunc: Function to decorate a node with attributes.
                          The function shall accept one `node` object as
                          argument and return the attributes.

            edgeattrfunc: Function to decorate a edge with attributes.
                          The function shall accept two `node` objects as
                          argument. The first the node and the second the child
                          and return the attributes.

            edgetypefunc: Function to which gives the edge type.
                          The function shall accept two `node` objects as
                          argument. The first the node and the second the child
                          and return the edge (i.e. '->').

            maxlevel (int): Limit export to this number of levels.

        >>> from anytree import Node
        >>> root = Node("root")
        >>> s0 = Node("sub0", parent=root, edge=2)
        >>> s0b = Node("sub0B", parent=s0, foo=4, edge=109)
        >>> s0a = Node("sub0A", parent=s0, edge="")
        >>> s1 = Node("sub1", parent=root, edge="")
        >>> s1a = Node("sub1A", parent=s1, edge=7)
        >>> s1b = Node("sub1B", parent=s1, edge=8)
        >>> s1c = Node("sub1C", parent=s1, edge=22)
        >>> s1ca = Node("sub1Ca", parent=s1c, edge=42)

        .. note:: If the node names are not unqiue, see :any:`UniqueDotExporter`.

        A directed graph:

        >>> from anytree.exporter import DotExporter
        >>> for line in DotExporter(root):
        ...     print(line)
        digraph tree {
            "root";
            "sub0";
            "sub0B";
            "sub0A";
            "sub1";
            "sub1A";
            "sub1B";
            "sub1C";
            "sub1Ca";
            "root" -> "sub0";
            "root" -> "sub1";
            "sub0" -> "sub0B";
            "sub0" -> "sub0A";
            "sub1" -> "sub1A";
            "sub1" -> "sub1B";
            "sub1" -> "sub1C";
            "sub1C" -> "sub1Ca";
        }

        The resulting graph:

        .. image:: ../static/dotexporter0.png

        An undirected graph:

        >>> def nodenamefunc(node):
        ...     return '%s:%s' % (node.name, node.depth)
        >>> def edgeattrfunc(node, child):
        ...     return 'label="%s:%s"' % (node.name, child.name)
        >>> def edgetypefunc(node, child):
        ...     return '--'
                >>> from anytree.exporter import DotExporter
        >>> for line in DotExporter(root, graph="graph",
        ...                             nodenamefunc=nodenamefunc,
        ...                             nodeattrfunc=lambda node: "shape=box",
        ...                             edgeattrfunc=edgeattrfunc,
        ...                             edgetypefunc=edgetypefunc):
        ...     print(line)
        graph tree {
            "root:0" [shape=box];
            "sub0:1" [shape=box];
            "sub0B:2" [shape=box];
            "sub0A:2" [shape=box];
            "sub1:1" [shape=box];
            "sub1A:2" [shape=box];
            "sub1B:2" [shape=box];
            "sub1C:2" [shape=box];
            "sub1Ca:3" [shape=box];
            "root:0" -- "sub0:1" [label="root:sub0"];
            "root:0" -- "sub1:1" [label="root:sub1"];
            "sub0:1" -- "sub0B:2" [label="sub0:sub0B"];
            "sub0:1" -- "sub0A:2" [label="sub0:sub0A"];
            "sub1:1" -- "sub1A:2" [label="sub1:sub1A"];
            "sub1:1" -- "sub1B:2" [label="sub1:sub1B"];
            "sub1:1" -- "sub1C:2" [label="sub1:sub1C"];
            "sub1C:2" -- "sub1Ca:3" [label="sub1C:sub1Ca"];
        }

        The resulting graph:

        .. image:: ../static/dotexporter1.png

        To export custom node implementations or :any:`AnyNode`, please provide a proper `nodenamefunc`:

        >>> from anytree import AnyNode
        >>> root = AnyNode(id="root")
        >>> s0 = AnyNode(id="sub0", parent=root)
        >>> s0b = AnyNode(id="s0b", parent=s0)
        >>> s0a = AnyNode(id="s0a", parent=s0)

        >>> from anytree.exporter import DotExporter
        >>> for line in DotExporter(root, nodenamefunc=lambda n: n.id):
        ...     print(line)
        digraph tree {
            "root";
            "sub0";
            "s0b";
            "s0a";
            "root" -> "sub0";
            "sub0" -> "s0b";
            "sub0" -> "s0a";
        }
        """
        self.node = node
        self.graph = graph
        self.name = name
        self.options = options
        self.indent = indent
        self.nodenamefunc = nodenamefunc
        self.nodeattrfunc = nodeattrfunc
        self.edgeattrfunc = edgeattrfunc
        self.edgetypefunc = edgetypefunc
        self.maxlevel = maxlevel
        self.include_nodes = include_nodes
        node_name_func = get_unique_node_names_func(node)
        self.include_node_names = [node_name_func(n) for n in self.include_nodes] if include_nodes else None

    def __iter__(self):
        # prepare
        indent = " " * self.indent
        nodenamefunc = self.nodenamefunc or self._default_nodenamefunc
        nodeattrfunc = self.nodeattrfunc or self._default_nodeattrfunc
        edgeattrfunc = self.edgeattrfunc or self._default_edgeattrfunc
        edgetypefunc = self.edgetypefunc or self._default_edgetypefunc
        return self.__iter(indent, nodenamefunc, nodeattrfunc, edgeattrfunc,
                           edgetypefunc)

    @staticmethod
    def _default_nodenamefunc(node):
        return node.name

    @staticmethod
    def _default_nodeattrfunc(node):
        return None

    @staticmethod
    def _default_edgeattrfunc(node, child):
        return None

    @staticmethod
    def _default_edgetypefunc(node, child):
        return "->"

    def __iter(self, indent, nodenamefunc, nodeattrfunc, edgeattrfunc, edgetypefunc):
        yield "{self.graph} {self.name} {{".format(self=self)
        for option in self.__iter_options(indent):
            yield option
        for node in self.__iter_nodes(indent, nodenamefunc, nodeattrfunc):
            yield node
        for edge in self.__iter_edges(indent, nodenamefunc, edgeattrfunc, edgetypefunc):
            yield edge
        yield "}"

    def __iter_options(self, indent):
        options = self.options
        if options:
            for option in options:
                yield "%s%s" % (indent, option)

    def __iter_nodes(self, indent, nodenamefunc, nodeattrfunc):
        for node in PreOrderIter(self.node, maxlevel=self.maxlevel):
            nodename = nodenamefunc(node)
            if self.include_nodes is not None and nodename not in self.include_node_names:
                continue
            nodeattr = nodeattrfunc(node)
            nodeattr = " [%s]" % nodeattr if nodeattr is not None else ""
            yield '%s"%s"%s;' % (indent, FilteredDotExporter.esc(nodename), nodeattr)

    def __iter_edges(self, indent, nodenamefunc, edgeattrfunc, edgetypefunc):
        maxlevel = self.maxlevel - 1 if self.maxlevel else None
        for node in PreOrderIter(self.node, maxlevel=maxlevel):
            nodename = nodenamefunc(node)
            if self.include_nodes is not None and nodename not in self.include_node_names:
                continue
            for child in node.children:
                childname = nodenamefunc(child)
                if self.include_nodes is not None and childname not in self.include_node_names:
                    continue
                edgeattr = edgeattrfunc(node, child)
                edgetype = edgetypefunc(node, child)
                edgeattr = " [%s]" % edgeattr if edgeattr is not None else ""
                yield '%s"%s" %s "%s"%s;' % (indent, FilteredDotExporter.esc(nodename), edgetype,
                                             FilteredDotExporter.esc(childname), edgeattr)

    def to_dotfile(self, filename):
        """
        Write graph to `filename`.

        >>> from anytree import Node
        >>> root = Node("root")
        >>> s0 = Node("sub0", parent=root)
        >>> s0b = Node("sub0B", parent=s0)
        >>> s0a = Node("sub0A", parent=s0)
        >>> s1 = Node("sub1", parent=root)
        >>> s1a = Node("sub1A", parent=s1)
        >>> s1b = Node("sub1B", parent=s1)
        >>> s1c = Node("sub1C", parent=s1)
        >>> s1ca = Node("sub1Ca", parent=s1c)

        >>> from anytree.exporter import DotExporter
        >>> DotExporter(root).to_dotfile("tree.dot")

        The generated file should be handed over to the `dot` tool from the
        http://www.graphviz.org/ package::

            $ dot tree.dot -T png -o tree.png
        """
        with codecs.open(filename, "w", "utf-8") as file:
            for line in self:
                file.write("%s\n" % line)

    def to_picture(self, filename):
        """
        Write graph to a temporary file and invoke `dot`.

        The output file type is automatically detected from the file suffix.

        *`graphviz` needs to be installed, before usage of this method.*
        """
        fileformat = os.path.splitext(filename)[1][1:]
        with NamedTemporaryFile("wb", delete=False) as dotfile:
            dotfilename = dotfile.name
            for line in self:
                dotfile.write(("%s\n" % line).encode("utf-8"))
            dotfile.flush()
            cmd = ["dot", dotfilename, "-T", fileformat, "-o", filename]
            check_call(cmd)
        try:
            os.remove(dotfilename)
        except Exception:  # pragma: no cover
            msg = 'Could not remove temporary file %s' % dotfilename
            logger.warning(msg)

    def to_source(self) -> Source:
        """
        Return the source code of the graph as a Source object.
        """
        return Source("\n".join(self), filename=self.name)

    @staticmethod
    def esc(value):
        """Escape Strings."""
        return _RE_ESC.sub(lambda m: r"\%s" % m.group(0), six.text_type(value))


def render_tree(root: Node, use_dot_exporter: bool = False,
                filename: str = "query_tree", only_nodes: List[Node] = None, show_in_console: bool = False,
                color_map: Optional[Callable[[Node], str]] = None,
                view: bool = False) -> None:
    """
    Render the tree using the console and optionally export it to a dot file.

    :param root: The root node of the tree.
    :param use_dot_exporter: Whether to export the tree to a dot file.
    :param filename: The name of the file to export the tree to.
    :param only_nodes: A list of nodes to include in the dot export.
    :param show_in_console: Whether to print the tree to the console.
    :param color_map: A function that returns a color for certain nodes.
    :param view: Whether to view the dot file in a viewer.
    :param use_legend: Whether to show the legend or not.
    """
    if not root:
        logger.warning("No nodes to render")
        return
    if show_in_console:
        for pre, _, node in RenderTree(root):
            if only_nodes is not None and node not in only_nodes:
                continue
            print(f"{pre}{node.weight if hasattr(node, 'weight') and node.weight else ''} {node.__str__()}")
    if use_dot_exporter:
        unique_node_names = get_unique_node_names_func(root)

        de = FilteredDotExporter(root,
                                 include_nodes=only_nodes,
                                 nodenamefunc=unique_node_names,
                                 edgeattrfunc=edge_attr_setter,
                                 nodeattrfunc=lambda node: \
                                     f'style=filled,'
                                     f' fillcolor={color_map(node) if color_map else getattr(node, "color", "white")}',
                                 )
        if view:
            de.to_source().view()
        else:
            filename = filename or "query_tree"
            de.to_dotfile(f"{filename}{'.dot'}")
            try:
                de.to_picture(f"{filename}{'.svg'}")
            except FileNotFoundError as e:
                logger.warning(f"{e}")
