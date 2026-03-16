from enum import Enum

class DType(str, Enum):
    INT = "int"
    FLOAT = "float"
    NUM = "num"         
    STR = "string"
    BOOL = "bool"
    DATETIME = "datetime"
    OTHER = "other"
