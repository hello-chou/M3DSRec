import argparse
from logging import getLogger, FileHandler
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color

from data.dataset import OurDataset
from trainer import OurTrainer

def get_logger_filename(logger):
    file_handler = next((handler for handler in logger.handlers if isinstance(handler, FileHandler)), None)
    if file_handler:
        filename = file_handler.baseFilename
        print(f"The log file name is {filename}")
    else:
        raise Exception("No file handler found in logger")
    return filename


def finetune(model, dataset,  props='props/FlowSASRec.yaml,props/finetune.yaml',  log_prefix="", **kwargs):
    props = props.split(',')
    print(props)

    config = Config(model=model, dataset=dataset, config_file_list=props, config_dict=kwargs)
    config['log_prefix'] = log_prefix

    init_seed(config['seed'], config['reproducibility'])

    init_logger(config)
    logger = getLogger()
    logger.info(config)


    dataset = OurDataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    if model=='alltoone':
        from alltoone import alltoone
        model = alltoone(config, train_data.dataset).to(config['device'])

    logger.info(model)


    # trainer loading and initialization
    trainer = OurTrainer(config, model)
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    logger_Filename = get_logger_filename(logger)
    logger.info(f"Write log to {logger_Filename}")

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='MISSRec', help='model name')
    parser.add_argument('-d', type=str, default='Pantry_mm_full', help='dataset name')
    parser.add_argument('-props', type=str, default='props/alltoone.yaml,props/finetune.yaml')
    parser.add_argument('-note', type=str, default='')
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(model=args.m,dataset=args.d, props=args.props,  log_prefix=args.note)
