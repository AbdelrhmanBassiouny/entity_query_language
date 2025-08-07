
class MultipleSolutionFound(Exception):
    def __init__(self, first_val, second_val):
        super(MultipleSolutionFound, self).__init__(f"Multiple solutions found, the first two are "
                                                    f"{first_val}\n{second_val}")
