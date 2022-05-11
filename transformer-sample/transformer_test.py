import argparse
import kserve
from typing import Dict
import logging
import pandas

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

def transform(instance):
    origin = instance[6]
    del instance[6]
    instance.insert(6, (origin == 1)*1.0)
    instance.insert(7, (origin == 2)*1.0)
    instance.insert(8, (origin == 3)*1.0)

    sr = pandas.Series(instance, index=['Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','USA','Europe','Japan'], dtype=float)

    df = pandas.read_csv("./train_stats.csv", index_col=0)

    normdata = (sr - df['mean']) / df['std']

    return normdata.values.tolist()



class NormTransformer(kserve.KFModel): 
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

        logging.info("MODEL NAME %s", name)
        logging.info("PREDICTOR URL %s", self.predictor_host)

        self.timeout = 100

    def preprocess(self, inputs: Dict) -> Dict:
        return {'instances': [transform(instance) for instance in inputs['instances']]}

    def postprocess(self, inputs:  Dict) -> Dict:
        return inputs


if __name__ == "__main__":   
    DEFAULT_MODEL_NAME = "model"

    parser = argparse.ArgumentParser(parents=[kserve.kfserver.parser])
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                            help='The name that the model is served under.')
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function", required=True
    )

    args, _ = parser.parse_known_args()

    transformer = NormTransformer(args.model_name, predictor_host=args.predictor_host)
    kfserver = kserve.KFServer()
    kfserver.start(models=[transformer])
