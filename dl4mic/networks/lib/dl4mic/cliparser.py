import argparse
import yaml

class ParserCreator(object):

    @classmethod
    def createArgumentParser(cls, parameterFile):
        parser = argparse.ArgumentParser()
        with open(parameterFile, 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
            for key in params.keys():
                parameterSet = params[key]
                for parameter in parameterSet:
                    if parameter['type'] == 'string' or parameter['type'] == 'file' or parameter['type'] == 'directory':
                        parser.add_argument('--' + parameter['name'], help=parameter['help'],
                                            default=parameter['default'])
                    if parameter['type'] == 'int':
                        parser.add_argument('--' + parameter['name'],
                                            help=parameter['help'],
                                            default=parameter['default'],
                                            type=int)
                    if parameter['type'] == 'float':
                        parser.add_argument('--' + parameter['name'],
                                            help=parameter['help'],
                                            default=parameter['default'],
                                            type=float)
                    if parameter['type'] == 'bool':
                        if parameter['value']==0:
                            storeParam = 'store_false'
                        else:
                            storeParam = 'store_true'
                        parser.add_argument('--' + parameter['name'],
                                            help=parameter['help'],
                                            default=parameter['default'],
                                            action=storeParam)
        return parser
