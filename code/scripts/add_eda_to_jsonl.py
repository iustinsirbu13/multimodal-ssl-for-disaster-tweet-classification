from modules.eda import eda
from argparse import ArgumentParser
import os
import jsonlines
from tqdm import tqdm


def sanity_checks(args):
    assert len(args.input_file) == len(args.output_file)
    for input_file, output_file in zip(args.input_file, args.output_file):
        assert os.path.isfile(input_file), f'{input_file} DOES NOT EXIST'
        assert args.override or not os.path.isfile(output_file), f'{output_file} ALREADY EXISTS'


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,  action='append')
    parser.add_argument('--output_file', type=str, required=True,  action='append')
    parser.add_argument('--num_augmentations', type=int, default=10)
    parser.add_argument('--override', action='store_true')
    args = parser.parse_args()

    sanity_checks(args)

    for input_file, output_file in zip(args.input_file, args.output_file):
        input_file, output_file = os.path.join(input_file), os.path.join(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f'\n Adding EDA to input file {input_file}')
        
        with jsonlines.open(input_file) as reader, jsonlines.open(output_file, 'w') as writer:
            for index, obj in tqdm(enumerate(reader)):
                try:
                    obj['eda_01'] = eda(obj['text'], 0.1, 0.1, 0.1, 0.1, args.num_augmentations)
                    obj['eda_sr'] = eda(obj['text'], 0.1, 0.0, 0.0, 0.0, args.num_augmentations, per_technique=True)
                    obj['eda_02'] = eda(obj['text'], 0.2, 0.2, 0.2, 0.2, args.num_augmentations)
                except:
                    print(f'Error at id {obj["id"]} with text: {obj["text"]}.')
                    obj['eda_01'] = [obj['text']]
                    obj['eda_02'] = [obj['text']]
                    obj['eda_sr'] = [obj['text']]
                writer.write(obj)

if __name__ == '__main__':
    main()
