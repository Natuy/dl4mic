import sys
from io import StringIO
sys.stderr = StringIO()
sys.path.append(r'../lib/')
from dl4mic.cliparser import ParserCreator
from dl4mic.n2v import N2VNetwork
sys.stderr = sys.__stderr__

def main(argv):
    parser = ParserCreator.createArgumentParser("./predict.yml")
    if len(argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args(argv[1:])
    print(args)
    n2v = N2VNetwork(args.name)
    n2v.setPath(args.baseDir)
    n2v.setTile((args.tileY, args.tileX))
    n2v.predict(args.dataPath, args.output)
    print("---predictions done---")


if __name__ == '__main__':
    main(sys.argv)
