"""
异常
"""


class IllegalArgumentException(Exception):
    """
    非法参数异常
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
