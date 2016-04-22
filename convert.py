#!/usr/bin/env python

import sys
import os
import logging
import argparse

import numpy as np

from kaffe import KaffeError
from kaffe.tensorflow import TensorFlowTransformer


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(module)s:%(funcName)s:%(lineno)d : %(message)s',
        level=logging.DEBUG)
    logger.info("running %s", " ".join(sys.argv))
    program = os.path.basename(sys.argv[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('data_path', help='Model data (.caffemodel) path')
    parser.add_argument('data_output_path', help='Converted data output path')
    parser.add_argument('code_output_path', nargs='?', help='Save generated source to this path')
    parser.add_argument('-p', '--phase', default='test', help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    try:
        transformer = TensorFlowTransformer(args.def_path, args.data_path, phase=args.phase)
        logger.info('Converting data...')
        data = transformer.transform_data()
        logger.info('Saving data...')
        with open(args.data_output_path, 'wb') as data_out:
            np.save(data_out, data)
        if args.code_output_path is not None:
            logger.info('Saving source...')
            with open(args.code_output_path, 'wb') as src_out:
                src_out.write(transformer.transform_source())
        logger.info('Done.')
    except KaffeError as err:
        logger.info('Error encountered: %s'%err)
        exit(-1)

    logger.info("finished running %s", program)


if __name__ == '__main__':
    main()
