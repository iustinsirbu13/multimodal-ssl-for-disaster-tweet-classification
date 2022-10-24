from modules.aidrtokenize import tokenize as clean_tweet
from argparse import ArgumentParser
import os
import jsonlines
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,  action='append')
    parser.add_argument('--output_file', type=str, required=True,  action='append')
    args = parser.parse_args()

    assert len(args.input_file) == len(args.output_file)

    for input_file, output_file in zip(args.input_file, args.output_file):
        input_file, output_file = os.path.join(input_file), os.path.join(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f'Solving for input file {input_file}\n')
        
        with jsonlines.open(input_file) as reader, jsonlines.open(output_file, 'w') as writer:
            for index, obj in tqdm(enumerate(reader)):
                obj['text'] = clean_tweet(obj['text'])
                writer.write(obj)


if __name__ == '__main__':
    main()
