from .node import AQPNode

class testnode(AQPNode):
    def __init__(self, id_: str, output_key: str = None, draw_options: dict = None, **kwargs):
        super().__init__(id_, output_key, draw_options, **kwargs)
        print("Test Node Init")
        self.type_ = "testnode"

    def execute(self, result: dict, **kwargs):
        super().execute(result, **kwargs)
        print("Test Node Execute")
        return 1