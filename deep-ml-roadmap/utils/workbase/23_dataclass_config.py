from dataclasses import dataclass
@dataclass
class Config:
    seed: int = 42
    test_size: float = 0.2
    model: str = "baseline"
print(Config())
