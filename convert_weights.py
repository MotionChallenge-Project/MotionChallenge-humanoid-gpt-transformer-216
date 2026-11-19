"""Extract and validate this checkpoint's GPU batch-inference weights."""
import argparse,json
from pathlib import Path
from torch_graph import convert,validate,sha


def prepare(root,runtime,source,output,artifacts,device):
    root,source,output=Path(root),Path(source),Path(output)
    outputs=runtime['tensor_outputs']
    report_path=output/'parity.json'
    if report_path.is_file() and (output/'graph.json').is_file() and (output/'weights.safetensors').is_file():
        graph=json.loads((output/'graph.json').read_text());report=json.loads(report_path.read_text())
        if (graph.get('source_sha256')==runtime['weights_sha256'] and
            graph.get('converter_sha256')==sha(root/'torch_graph.py') and
            graph.get('weights_sha256')==sha(output/'weights.safetensors') and
            report.get('status')=='passed' and report.get('graph_sha256')==sha(output/'graph.json')):
            return output
    convert(source,output,runtime['weights_sha256'],outputs)
    report=validate(source,output,device=device,batches=(1,7,40),atol=5e-5)
    # The conversion cache carries its own validation receipt.
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    runtime=json.loads((root/'runtime.mjwarp.json').read_text())
    print(prepare(root,runtime,args.source,args.output,{},args.device))


if __name__=='__main__':main()
