import os

import pytest

try:
    from rustworkx_utils import GraphVisualizer
except ImportError:
    GraphVisualizer = None

from .datasets import Drawer, Handle, FixedConnection, Body, Container, PrismaticConnection, RevoluteConnection, View, \
    Door, Wardrobe
from entity_query_language import entity, rule_mode, infer, HasType, symbolic_mode, Add, alternative

@pytest.mark.skipif(GraphVisualizer is None, reason="requires rustworkx_utils")
def test_render_rx_graph_as_igraph_simple(handles_and_containers_world):
    world = handles_and_containers_world
    with rule_mode():
        fixed_connection = FixedConnection(world=world)
        container = fixed_connection.parent
        handle = fixed_connection.child
        rule = infer(entity(Drawer(handle=handle, container=container, world=world),
                               HasType(handle, Handle)))
    drawers = list(rule.evaluate())
    if os.path.exists("pdf_graph.pdf"):
        os.remove("pdf_graph.pdf")
    rule.visualize()
    assert os.path.exists("pdf_graph.pdf")


@pytest.mark.skipif(GraphVisualizer is None, reason="requires rustworkx_utils")
def test_render_rx_graph_as_igraph_complex(doors_and_drawers_world):
    world = doors_and_drawers_world
    with symbolic_mode():
        body = Body(world=world)
        handle = Handle(world=world)
        container = Container(world=world)
        fixed_connection = FixedConnection(parent=body, child=handle, world=world)
        prismatic_connection = PrismaticConnection(child=body, world=world)
        revolute_connection = RevoluteConnection(parent=body, child=handle, world=world)
        rule = infer(views := View(), fixed_connection, prismatic_connection)

    with rule_mode(rule):
        Add(views, Drawer(handle=handle, container=body, world=world))
        with alternative(revolute_connection):
            Add(views, Door(handle=handle, body=body, world=world))
        with alternative(fixed_connection,
                         body == revolute_connection.child,
                         container == revolute_connection.parent,
                         revolute_connection.world == world):
            Add(views, Wardrobe(handle=handle, body=body, container=container, world=world))
    results = list(rule.evaluate())
    if os.path.exists("pdf_graph.pdf"):
        os.remove("pdf_graph.pdf")
    rule.visualize()
    assert os.path.exists("pdf_graph.pdf")

