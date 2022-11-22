from doctest import Example
from logging import exception, raiseExceptions
import logging
import os
import time
from typing_extensions import Required

from kubernetes import client 
from kserve import *

from flask import Flask
from flask import Response
from flask import request
from flask import jsonify
from flask_restx import Api, Resource, fields, reqparse

from dataclasses import dataclass


#logger settings
logging.basicConfig(level=logging.DEBUG,
                   format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                   datefmt='%Y-%m-%d %H:%M:%S',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger()

#Flask Settings
app = Flask(__name__)

#restX Settings
api = Api(app, version='1.0', title='KoreServe Inference Service API',
    description='A Wraped Kserve python SDK To REST-API',
)
ns = api.namespace('', description='REST-API operations')

#Kserve Settings
target_namespace = utils.get_default_target_namespace()
api_version      = constants.KSERVE_GROUP + '/' + constants.KSERVE_V1BETA1


#OS Env Settings
KSERVE_API_K8S_CONFIG_FILE               = os.environ.get('KSERVE_API_K8S_CONFIG_FILE')
KSERVE_API_DEFAULT_AWS_ACCESS_KEY_ID     = os.environ.get('KSERVE_API_DEFAULT_AWS_ACCESS_KEY_ID')
KSERVE_API_DEFAULT_AWS_SECRET_ACCESS_KEY = os.environ.get('KSERVE_API_DEFAULT_AWS_SECRET_ACCESS_KEY')
KSERVE_API_DEFAULT_STORAGE_ENDPOINT      = os.environ.get('KSERVE_API_DEFAULT_STORAGE_ENDPOINT')
KSERVE_API_LOGGERURL                     = os.environ.get('KSERVE_API_LOGGERURL')


#지정된 K8S Config 파일이 없으면 POD의 SA 실행권한으로 구동
if(KSERVE_API_K8S_CONFIG_FILE != "" and KSERVE_API_K8S_CONFIG_FILE != None):
    KServe = KServeClient(config_file=KSERVE_API_K8S_CONFIG_FILE)
else:
    KServe = KServeClient()




#######################################################################
# restX Input Model                                                         #
#######################################################################
predictor_resource_detail_model = api.model("ResourceDetail", {
                    "cpu": fields.String(example="2"),
                    "memory": fields.String(example="2Gi")
})

predictor_resource_model = api.model("Resource", {
    "requests": fields.Nested(predictor_resource_detail_model),
    "limits": fields.Nested(predictor_resource_detail_model)
})

predictor_modelspec_model = api.model('ModelSpec',{
    "modelframwwork": fields.String(enum=['tensorflow', 'sklearn', 'xgboost', 'lightgbm', 'onnx', 'pytorch', 'triton'], example="tensorflow"),
    "storageuri": fields.String(example="s3://testmodel/mpg2"),
    "runtime_version": fields.String(example="1.14.0")
})

predictor_model =  api.model('Predictor',{
    "modelspec": fields.Nested(predictor_modelspec_model),
    "logger": fields.String(enum=['all', 'request', 'response'], example="all"),
    "resource": fields.Nested(predictor_resource_model),
    "min_replicas": fields.Integer(example=1),
    "max_replicas": fields.Integer(example=1),
    "canary_traffic_percent": fields.Integer(required=False, example="100"),
})

transformer_model =  api.model('Transformer',{
    "image": fields.String(example="192.168.88.155/mpg-sample/transformer:v8"),
    "logger": fields.String(enum=['all', 'request', 'response']),
    "resource": fields.Nested(predictor_resource_model)
})

storage_model =  api.model('Storage',{
    "storage_credential": fields.String,  
    "storage_endpoint"  : fields.String
})


inference_service_model = api.model("InferenceService", {
                    "namespace": fields.String(required=True, example="default"),
                    "inferencename": fields.String(required=True, example="mpg-sample"),
                    "predictor": fields.Nested(predictor_model, required=True),
                    "transformer": fields.Nested(transformer_model),
                    #"storage": fields.Nested(storage_model),
                    "autoscaling_target_count": fields.Integer(example=10)
})


#######################################################################
# restx Output Model                                                         #
#######################################################################
output_post_model = api.model('OutPutModel', {
    'message': fields.String,
    'inferencename': fields.String,
    'revision': fields.String,
    'url': fields.String,
    'data': fields.String
})

output_get_model = api.model('OutPutModel', {
    'message': fields.String,
    'inferencename': fields.String,
    'data': fields.String
})

output_delete_model = api.model('OutPutModel', {
    'message': fields.String,
    'inferencename': fields.String,
})



##global error definition
@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({'massage': 'Bad request!'}), 400

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'massage': 'Page not found'}), 404

@app.errorhandler(500)
def page_not_found(e):
    return jsonify({'massage': 'Internal server error'}),  500




############################################################
# HTTP Routing
############################################################

#swagger document annotaion
@ns.route("/inference-service/<string:inferencename>")
@ns.param('inferencename', 'Kserve Inferencename')
@ns.param('namespace', 'The K8S Cluster Namespace identifier', required = 'true')
#class
class KserveAPI(Resource):
    #swagger document annotaion
    #response model mashalling
    @ns.marshal_with(output_get_model, code=200, skip_none=True) 
    @api.response(400, 'NotFound inference name')
    @api.response(500, 'Internal server error')
    #method
    def get(self, inferencename):
        parser = reqparse.RequestParser()
        parser.add_argument('namespace', required=True, type=str, location='args')
        args = parser.parse_args()

        inference_name = inferencename
        target_namespace = args.get('namespace')

        result = get_inference(inference_name, target_namespace)

        if(result == "NotFound"):
            return {"message":"NotFound inference name"}, 400
        elif(result == "Error"):
            return {"message":"Internal server error"}, 500    
        else:
            return {"message":"Success", "inferencename": inference_name, "data":result}, 200
 
    @ns.marshal_with(output_delete_model, code=200, skip_none=True)  
    @api.response(400, 'NotFound inference name')
    @api.response(500, 'Internal server error')
    def delete(self, inferencename):
        parser = reqparse.RequestParser()
        parser.add_argument('namespace', required=True, type=str, location='args')
        args = parser.parse_args()

        inference_name = inferencename
        target_namespace = args.get('namespace')

        result = delete_inference(inference_name, target_namespace)
        if(result == "NotFound"):
            return {"message":"NotFound inference name"}, 400
        elif(result == "Error"):
            return {"message":"Internal server error"}, 500    
        else:
            return {"message":result, "inferencename": inference_name}, 200        

@ns.route("/inference-service")  
class KserveAPIPost(Resource):  
    @ns.expect(inference_service_model, validate=True) 
    @ns.marshal_with(output_post_model, code=201, skip_none=True)
    @api.response(400, 'Already exist inference name')
    @api.response(500, 'Inference service create fail')  
    def post(self):
        #parsing request
        args = api.payload

        # target_namespace = args.get('namespace')
        # inference_name   = args.get('inferencename')

        predictor, transformer, model_storage, inference_service = parsing_post_data(args)
        
        #call create inference
        if(get_inference(inference_service.inference_name, inference_service.namespace) != "NotFound"):
            return {"message":"Already exist inference name"}, 400

        result, url = create_inference(predictor, inference_service, transformer, model_storage)     
        if(result):
            return {"message":"Success", "inferencename":inference_service.inference_name, "revision":"00001", "url":url}, 201
        else:
            return {"message":"Inference service create fail"}, 500

    @ns.expect(inference_service_model, validate=True) 
    @ns.marshal_with(output_post_model, code=200, skip_none=True) 
    @api.response(400, 'NotFound inference name')
    @api.response(500, 'Inference service replace fail')  
    def put(self):
        #parsing request
        args = api.payload
        
        # target_namespace = args.get('namespace')
        # inference_name   = args.get('inferencename')

        predictor, transformer, model_storage, inference_service = parsing_post_data(args)

        #call create inference
        if(get_inference(inference_service.inference_name, inference_service.namespace) == "NotFound"):
            return {"message":"NotFound inference name"}, 400

        result, revision, url=replace_inference(predictor, inference_service, transformer, model_storage)    
        if(revision):
            return {"message":"Success", "inferencename":inference_service.inference_name, "revision":revision, "url":url}, 200
        else:
            return {"message":"Inference service replace fail"}, 500 

 




############################################################
# Domain Logic
############################################################
@dataclass
class Predictor:
    model_framwwork            : str
    runtime_version            : str  
    storage_uri                : str
    logger                     : str
    resource_limits_cpu        : str
    resource_limits_memory     : str
    resource_limits_gpu        : str
    resource_requests_cpu      : str
    resource_requests_memory   : str
    max_replicas               : int
    min_replicas               : int
    canary_traffic_percent     : int


@dataclass
class Transformer:
    image                      : str  
    logger                     : str
    resource_limits_cpu        : str
    resource_limits_memory     : str
    resource_limits_gpu        : str
    resource_requests_cpu      : str
    resource_requests_memory   : str

@dataclass
class ModelStorage:
    storage_credential: str  
    storage_endpoint  : str

@dataclass
class InferenceService:
    namespace               : str  
    inference_name           : str
    autoscaling_target_count: int 

def create_inference(predictor, inference_service, transformer, model_storage):
    
    try:
        if(make_credential(model_storage, predictor, inference_service)):
            print("areyouok?1")
            isvc = make_spec(predictor, inference_service, transformer)
            print("areyouok?2")
        else:
            return False

        logger.info(isvc)

        KServe.create(isvc, inference_service.namespace)
        KServe.wait_isvc_ready(inference_service.inference_name, namespace=inference_service.namespace, watch=False, timeout_seconds=600)
        inference_check = (get_inference(inference_service.inference_name, inference_service.namespace))
        url = inference_check['status']['url']

        return True, url

    except Exception as err:
        logger.exception("create_inference exception:" + str(err))
        delete_inference(inference_service.inference_name, inference_service.namespace)
        return False 



def replace_inference(predictor, inference_service, transformer, model_storage):
    try:
        if(make_credential(model_storage, predictor, inference_service)):
            isvc = make_spec(predictor, inference_service, transformer)
        else:
            return False

        print(isvc)
        #
        KServe.replace(inference_service.inference_name, isvc, inference_service.namespace)
        KServe.wait_isvc_ready(inference_service.inference_name, namespace=inference_service.namespace, watch=True, timeout_seconds=600)
        
        time.sleep(5)
        inference_check = (get_inference(inference_service.inference_name, inference_service.namespace))
        
        revision = inference_check['status']['components']['predictor']['latestCreatedRevision']
        revision = revision[len(revision)-5:len(revision)]
        url = inference_check['status']['url']
        
        return True, revision, url

    except Exception as err:
        logger.exception("replace_inference exception:" + str(err)) 
        return False 


def get_inference(inference_name, target_namespace):
    print("get target_namespace:" + target_namespace)
    try:
        result = KServe.get(inference_name, target_namespace, watch=False, timeout_seconds=120)
        return result
    except RuntimeError as err:
        if(str(err).find("(404)") >= 0):
            return "NotFound"
        else:
            logger.exception("get_inference runtime exception:" + str(err))
            return "Error"    
          
    except Exception as err:
        logger.exception("get_inference exception:" + str(err)) 
        return "Error"

    
def delete_inference(inference_name, target_namespace):
    try:
        KServe.delete(inference_name, target_namespace)
        return "Success"
    except RuntimeError as err:
        if(str(err).find("(404)") >= 0):
            return "NotFound"
        else:
            logger.exception("delete_inference runtime exception:" + str(err))
            return "Error"    
          
    except Exception as err:
        logger.exception("delete_inference exception:" + str(err))
        return "Error"




def parsing_post_data(request_data):
    print(request_data)
    namespace = request_data.get('namespace')
    inference_name = request_data.get('inferencename')
    autoscaling_target_count = request_data.get('autoscaling_target_count')  
    resource_limits_cpu = None
    resource_limits_memory = None
    resource_limits_gpu =   None
    resource_requests_cpu = None
    resource_requests_memory = None

    #Autoscaling할 동시접속 타켓 카운트 초기값 설정
    # if(autoscaling_target_count == None):
    #     autoscaling_target_count = 10

    inference_service = InferenceService(namespace,
                                        inference_name,
                                        autoscaling_target_count)

    if(request_data.get('predictor') != None):
        model_framwwork  = request_data.get('predictor').get('modelspec').get('modelframwwork')
        runtime_version  = request_data.get('predictor').get('modelspec').get('runtime_version')
        storage_uri      = "s3://" + request_data.get('predictor').get('modelspec').get('storageuri')
        min_replicas     = request_data.get('predictor').get('min_replicas')
        max_replicas     = request_data.get('predictor').get('max_replicas')
        logger = request_data.get('predictor').get('logger')
        canary_traffic_percent = request_data.get('predictor').get('canary_traffic_percent')

        if(request_data.get('predictor').get('resource') != None):
            if(request_data.get('predictor').get('resource').get('limits') != None):
                resource_limits_cpu = request_data.get('predictor').get('resource').get('limits').get('cpu')
                resource_limits_memory = request_data.get('predictor').get('resource').get('limits').get('memory')
                resource_limits_gpu = request_data.get('predictor').get('resource').get('limits').get('gpu')

            if(request_data.get('predictor').get('resource').get('requests') != None):    
                resource_requests_cpu = request_data.get('predictor').get('resource').get('requests').get('cpu')
                resource_requests_memory = request_data.get('predictor').get('resource').get('requests').get('memory')
  
        predictor = Predictor(model_framwwork, 
                              runtime_version, 
                              storage_uri, 
                              logger, 
                              resource_limits_cpu, 
                              resource_limits_memory, 
                              resource_limits_gpu, 
                              resource_requests_cpu, 
                              resource_requests_memory,
                              min_replicas,
                              max_replicas,
                              canary_traffic_percent      
                )
    else:
         predictor = None                          
    
    if(request_data.get('transformer') != None):
        tr_image                    = request_data.get('transformer').get('image')
        tr_logger                   = request_data.get('transformer').get('logger')
        tr_resource_limits_cpu      = request_data.get('transformer').get('resource').get('limits').get('cpu')
        tr_resource_limits_memory   = request_data.get('transformer').get('resource').get('limits').get('memory')
        tr_resource_limits_gpu      = request_data.get('transformer').get('resource').get('limits').get('gpu')
        tr_resource_requests_cpu    = request_data.get('transformer').get('resource').get('limits').get('cpu')
        tr_resource_requests_memory = request_data.get('transformer').get('resource').get('limits').get('memory')

        transformer = Transformer(tr_image,                   
                                  tr_logger,                  
                                  tr_resource_limits_cpu,     
                                  tr_resource_limits_memory,  
                                  tr_resource_limits_gpu,     
                                  tr_resource_requests_cpu,   
                                  tr_resource_requests_memory)
    else:
         transformer = None

    storage_credential = request_data.get('storage_credential')
    storage_endpoint = request_data.get('storage_endpoint')    
    
    model_storage = ModelStorage(storage_credential, 
                                 storage_endpoint)

    if(storage_credential == None and storage_endpoint == None):
        model_storage.storage_credential = "[default]" + "\n" + "AWS_ACCESS_KEY_ID:" + KSERVE_API_DEFAULT_AWS_ACCESS_KEY_ID + "\n" + "AWS_SECRET_ACCESS_KEY:" + KSERVE_API_DEFAULT_AWS_SECRET_ACCESS_KEY
        model_storage.storage_endpoint = KSERVE_API_DEFAULT_STORAGE_ENDPOINT
    




    return predictor, transformer, model_storage, inference_service


def make_credential(model_storage, predictor, inference_service):
    #make credential & spec string
    # if(predictor.storage_uri.find("s3:") >= 0):
        with open('/tmp/credential.txt', 'w') as credential_file:
            credential_file.write(model_storage.storage_credential)

        KServe.set_credentials(storage_type='s3', namespace=inference_service.namespace, credentials_file=credential_file.name, service_account=inference_service.inference_name,  s3_profile='default', s3_endpoint=model_storage.storage_endpoint, s3_region='us-west-2', s3_use_https='0', s3_verify_ssl='0') 
        
        credential_file.close()
        return True
    # else:
        # return True


def make_spec(predictor, inference_service, transformer):
    if(predictor.model_framwwork == "tensorflow"):
        model_framwwork_spec_str =  "tensorflow=V1beta1TFServingSpec"
    elif(predictor.model_framwwork == "sklearn"):
        model_framwwork_spec_str =  "sklearn=V1beta1SKLearnSpec"
    elif(predictor.model_framwwork == "xgboost"):
        model_framwwork_spec_str =  "xgboost=V1beta1XGBoostSpec"
    elif(predictor.model_framwwork == "lightgbm"):
        model_framwwork_spec_str =  "lightgbm=V1beta1LightGBMSpec"    
    elif(predictor.model_framwwork == "onnx"):
        model_framwwork_spec_str =  "onnx=V1beta1ONNXRuntimeSpec"    
    elif(predictor.model_framwwork == "pytorch"):
        model_framwwork_spec_str =  "pytorch=V1beta1TorchServeSpec"      
    elif(predictor.model_framwwork == "triton"):
        model_framwwork_spec_str =  "triton=V1beta1TritonSpec"  
    else:
        return  False

    #make resources spec string
    #kserve 0.7 default request&limits cpu=1 memory=2Gi 
    if(predictor.resource_limits_cpu != None 
       or predictor.resource_limits_memory != None 
       or predictor.resource_limits_gpu != None ):
        limits = []
        limits.append("limits={")
        if(predictor.resource_limits_cpu != None):
            limits.append("'cpu':'" + predictor.resource_limits_cpu + "'")
        if(predictor.resource_limits_memory != None):   
            limits.append(", 'memory':'" + predictor.resource_limits_memory + "'")
        if(predictor.resource_limits_gpu   != None): 
            limits.append(", 'nvidia.com/gpu':" + predictor.resource_limits_gpu + "'")
        limits.append("}")
        limits = ''.join(limits)
    else:
        limits = ""

    if(predictor.resource_requests_cpu != None
       or predictor.resource_requests_memory != None):
        requests = []
        requests.append("requests={")
        if(predictor.resource_requests_cpu   != None):
            requests.append("'cpu':'" + predictor.resource_requests_cpu + "'")
        if(predictor.resource_requests_memory   != None):   
            requests.append(", 'memory':'" + predictor.resource_requests_memory + "'")
        requests.append("}")
        requests = ''.join(requests)
    else:
        requests = ""    
    
    predictor_resource_spec_str = ""
    if(limits != "" or requests != ""):
        predictor_resource_spec_str = []
        predictor_resource_spec_str.append(", resources=client.V1ResourceRequirements(")
        if(limits != ""):
            predictor_resource_spec_str.append(limits)
        if(requests != ""):
            if(limits != ""):
                predictor_resource_spec_str.append(", ")
            predictor_resource_spec_str.append(requests)
        predictor_resource_spec_str.append(") ")
        predictor_resource_spec_str = ''.join(predictor_resource_spec_str)


    #make runtime version spec string
    if(predictor.runtime_version != None):
        runtime_verion_spec_str = ", runtime_version=\'" + predictor.runtime_version + "\'" 
    else:
        runtime_verion_spec_str = ""

    #make max replicas spec string
    if(predictor.max_replicas != None):
        max_replicas_spec_str = ", max_replicas=" + str(predictor.max_replicas)
    else:
        max_replicas_spec_str = ""

    #make min replicas spec string
    if(predictor.max_replicas != None):
        min_replicas_spec_str = ", min_replicas=" + str(predictor.min_replicas)
    else:
        min_replicas_spec_str = ""

    #make logger spec string
    if(predictor.logger != None):
        predictor_logger_spec_str = ", logger=V1beta1LoggerSpec(mode=\'" + predictor.logger + "\',url=\'" +KSERVE_API_LOGGERURL + "\')"
    else:
        predictor_logger_spec_str = ""

    #make canary spec string
    if(predictor.logger != None):
        predictor_canary_spec_str = ", canary_traffic_percent=" + str(predictor.canary_traffic_percent)
    else:
        predictor_canary_spec_str = ""



    predictor_spec_str = "predictor=V1beta1PredictorSpec(" + "service_account_name=\'" + inference_service.inference_name + "\', " + model_framwwork_spec_str + "(storage_uri='" + predictor.storage_uri + "'" + \
                                                             predictor_resource_spec_str + \
                                                             runtime_verion_spec_str + \
                                                           ")"  + \
                                                          predictor_logger_spec_str + \
                                                          min_replicas_spec_str + max_replicas_spec_str + \
                                                          predictor_canary_spec_str + ")"


    #make transformer spec string
    if(transformer != None):
        if(transformer.logger != None):
            transformer_logger_spec_str = ", logger=V1beta1LoggerSpec(mode=\'" + transformer.logger + "\',url=\'" + KSERVE_API_LOGGERURL + "\')"
        else:
            transformer_logger_spec_str = ""

        if(transformer.resource_limits_cpu != None 
        or transformer.resource_limits_memory != None 
        or transformer.resource_limits_gpu != None ):
            limits = []
            limits.append("limits={")
            if(transformer.resource_limits_cpu != None):
                limits.append("'cpu':'" + transformer.resource_limits_cpu + "'")
            if(transformer.resource_limits_memory != None):   
                limits.append(", 'memory':'" + transformer.resource_limits_memory + "'")
            if(transformer.resource_limits_gpu   != None): 
                limits.append(", 'nvidia.com/gpu':" + transformer.resource_limits_gpu + "'")
            limits.append("}")
            limits = ''.join(limits)
        else:
            limits = ""

        if(transformer.resource_requests_cpu != None
        or transformer.resource_requests_memory != None):
            requests = []
            requests.append("requests={")
            if(transformer.resource_requests_cpu   != None):
                requests.append("'cpu':'" + transformer.resource_requests_cpu + "'")
            if(transformer.resource_requests_memory   != None):   
                requests.append(", 'memory':'" + transformer.resource_requests_memory + "'")
            requests.append("}")
            requests = ''.join(requests)
        else:
            requests = ""    
        
        if(limits != "" or requests != ""):
            transformer_resource_spec_str = []
            transformer_resource_spec_str.append(", resources=client.V1ResourceRequirements(")
            if(limits != ""):
                transformer_resource_spec_str.append(limits)
            if(requests != ""):
                if(limits != ""):
                    transformer_resource_spec_str.append(", ")
                transformer_resource_spec_str.append(requests)
            transformer_resource_spec_str.append(") ")
            transformer_resource_spec_str = ''.join(transformer_resource_spec_str)

        transformer_spec_str = ", transformer=V1beta1TransformerSpec(" + "service_account_name=\'" + inference_service.inference_name + "\'," \
                                                                    + "containers=[client.V1Container(name='transformer', " + \
                                                                        "image='" + transformer.image + "', " +  \
                                                                        "env=[client.V1EnvVar(name='STORAGE_URI', value='" + predictor.storage_uri + "')]" + \
                                                                        transformer_resource_spec_str + \
                                                                      ")]" + transformer_logger_spec_str + \
                                                                    ")"
    else:
        transformer_spec_str = ""



    #default_model_spec = "V1beta1InferenceServiceSpec(predictor=V1beta1PredictorSpec(" + model_framwwork_spec_str + "(storage_uri='" + storage_uri + "'" + ")" + logger_spec_str + "))"
    default_model_spec = "V1beta1InferenceServiceSpec(" + predictor_spec_str + transformer_spec_str + ")"
    print(default_model_spec)

    #make autoscaling spec string
    if(inference_service.autoscaling_target_count != None):
        autoscaling_spec_str = "{\'autoscaling.knative.dev/target\':\'" + str(inference_service.autoscaling_target_count) + "\'}"
    else:
        autoscaling_spec_str = "{}"
    #make iference service spec
    isvc = V1beta1InferenceService(api_version=constants.KSERVE_V1BETA1,
                                  kind=constants.KSERVE_KIND,
                                  metadata=client.V1ObjectMeta(name=inference_service.inference_name, namespace=inference_service.namespace, annotations=eval(autoscaling_spec_str)),
                                  spec=eval(default_model_spec))
    print(isvc)
    return isvc     



############################################################
# Main
############################################################
if __name__ == '__main__':
    app.run()
    logger.info("Kserve-API Server Start!!")    