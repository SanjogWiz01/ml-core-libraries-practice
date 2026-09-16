import argparse
parser = argparse.ArgumentParser(description="ML utility demo")
parser.add_argument("--model", default="baseline")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
print(vars(args))
