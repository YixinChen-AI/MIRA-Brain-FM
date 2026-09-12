"""Command-line ROI extraction."""
import argparse
from omnimira.pretrained import from_pretrained

def main():
    p=argparse.ArgumentParser(prog="omnimira",description="Extract atlas-indexed brain ROI features")
    p.add_argument("input"); p.add_argument("output"); p.add_argument("--modality",required=True,choices=("t1","av45","fdg","ct")); p.add_argument("--device",default="cpu"); p.add_argument("--checkpoint")
    a=p.parse_args(); from_pretrained(a.device,a.checkpoint).save(a.input,a.output,a.modality); print(a.output)
if __name__=="__main__": main()
