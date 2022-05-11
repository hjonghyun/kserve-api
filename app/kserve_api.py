from logging import exception, raiseExceptions
import logging
import os
import time

from kubernetes import client 
from kserve import KServeClient
from kserve import constants
from kserve import utils
from kserve import V1beta1InferenceService
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1PredictorSpec
from kserve import V1beta1TFServingSpec
from kserve import V1beta1LoggerSpec
from kserve import V1beta1TransformerSpec

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
# try:
#     app.config.from_envvar('APP_CONFIG_FILE')
# except Exception as err:
#     logger.exception("OS Env Config Error:" + str(err))

app.config.from_prefixed_env()

#restX Settings
api = Api(app, version='1.0', title='KoreServe Inference Service API',
    description='A Wraped Kserve python SDK To REST-API',
)
ns = api.namespace('', description='REST-API operations')

#Kserve Settings
target_namespace = utils.get_default_target_namespace()
api_version = constants.KSERVE_GROUP + '/' + constants.KSERVE_V1BETA1

if(os.environ['KSERVE_API_K8S_CONFIG_FILE'] != None):
    KServe = KServeClient(config_file=os.environ['KSERVE_API_K8S_CONFIG_FILE'])
else:
    KServe = KServeClient()




#######################################################################
# restX Input Model                                                         #
#######################################################################
predictor_resource_detail_model = api.model("ResourceDetail", {
                    "cpu": fields.String,
                    "memory": fields.String
})

predictor_resource_model = api.model("Resource", {
    "requests": fields.Nested(predictor_resource_detail_model),
    "limits": fields.Nested(predictor_resource_detail_model)
})

predictor_modelspec_model = api.model('ModelSpec',{
    "modelframwwork": fields.String,
    "storageuri": fields.String,
    "runtime_version": fields.String
})

predictor_model =  api.model('Predictor',{
    "modelspec": fields.Nested(predictor_modelspec_model),
    "logger": fields.String,
    "resource": fields.Nested(predictor_resource_model)
})

transformer_model =  api.model('Transformer',{
    "image": fields.String,
    "logger": fields.String,
    "resource": fields.Nested(predictor_resource_model)
})

storage_model =  api.model('Storage',{
    "storage_credential": fields.String,  
    "storage_endpoint"  : fields.String
})


inference_service_model = api.model("InferenceService", {
                    "namespace": fields.String(required=True),
                    "inferencename": fields.String(required=True),
                    "predictor": fields.Nested(predictor_model, required=True),
                    "transformer": fields.Nested(transformer_model),
                    "storage": fields.Nested(storage_model)
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
@ns.route("/inference-service/<string:inferencename>")
@ns.param('inferencename', 'Kserve Inferencename')
@ns.param('namespace', 'The K8S Cluster Namespace identifier')
class KserveAPI(Resource):
    @ns.marshal_with(output_get_model, code=200, skip_none=True) 
    @api.response(400, 'NotFound inference name')
    @api.response(500, 'Internal server error')
    def get(self, inferencename):
        sum_parser = ns.parser()
        args = sum_parser.parse_args()

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
        sum_parser = ns.parser()
        args = sum_parser.parse_args()

        target_namespace = args.get('namespace')
        inference_name = inferencename

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

        target_namespace = args.get('namespace')
        inference_name   = args.get('inferencename')

        predictor, transformer, model_storage = parsing_post_data(args)
        
        #call create inference
        if(get_inference(inference_name, target_namespace) != "NotFound"):
            return {"message":"Already exist inference name"}, 400

        result, url = create_inference(predictor, inference_name, target_namespace, transformer, model_storage)     
        if(result):
            return {"message":"Success", "inferencename":inference_name, "revision":"00001", "url":url}, 201
        else:
            return {"message":"Inference service create fail"}, 500

    @ns.expect(inference_service_model, validate=True) 
    @ns.marshal_with(output_post_model, code=200, skip_none=True) 
    @api.response(400, 'NotFound inference name')
    @api.response(500, 'Inference service replace fail')  
    def put(self):
        #parsing request
        args = api.payload
        
        target_namespace = args.get('namespace')
        inference_name   = args.get('inferencename')

        predictor, transformer, model_storage = parsing_post_data(args)

        #call create inference
        if(get_inference(inference_name, target_namespace) == "NotFound"):
            return {"message":"NotFound inference name"}, 400

        result, revision, url=replace_inference(predictor, inference_name, target_namespace, transformer, model_storage)    
        if(revision):
            return {"message":"Success", "inferencename":inference_name, "revision":revision, "url":url}, 200
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

def create_inference(predictor, inference_name, target_namespace, transformer, model_storage):
    try:
        if(make_credential(model_storage, predictor)):
            isvc = make_spec(predictor, inference_name, transformer)
        else:
            return False

        logger.info(isvc)

        KServe.create(isvc, target_namespace)
        KServe.wait_isvc_ready(inference_name, namespace=target_namespace, watch=False, timeout_seconds=600)
        inference_check = (get_inference(inference_name, target_namespace))
        print(inference_check)
        url = inference_check['status']['url']
        print(url)
        return True, url

    except Exception as err:
        logger.exception("create_inference exception:" + str(err))
        delete_inference(inference_name, target_namespace)
        return False 



def replace_inference(predictor, inference_name, target_namespace, transformer, model_storage):
    try:
        if(make_credential(model_storage, predictor)):
            isvc = make_spec(predictor, inference_name, transformer)
        else:
            return False

        print(isvc)
        #
        KServe.replace(inference_name, isvc, target_namespace)
        KServe.wait_isvc_ready(inference_name, namespace=target_namespace, watch=True, timeout_seconds=600)
        
        time.sleep(5)
        inference_check = (get_inference(inference_name, target_namespace))
        
        revision = inference_check['status']['components']['predictor']['latestCreatedRevision']
        revision = revision[len(revision)-5:len(revision)]
        url = inference_check['status']['url']
        
        return True, revision, url

    except Exception as err:
        logger.exception("replace_inference exception:" + str(err)) 
        return False 



def get_inference(inference_name, target_namespace):
    try:
        result = KServe.get(inference_name, namespace=target_namespace, watch=False, timeout_seconds=120)
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
        KServe.delete(inference_name, namespace=target_namespace)
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
    if(request_data.get('predictor') != None):
        model_framwwork  = request_data.get('predictor').get('modelspec').get('modelframwwork')
        runtime_version  = request_data.get('predictor').get('modelspec').get('runtime_version')
        storage_uri      = request_data.get('predictor').get('modelspec').get('storageuri')
        logger = request_data.get('predictor').get('logger')
        resource_limits_cpu = request_data.get('predictor').get('resource').get('limits').get('cpu')
        resource_limits_memory = request_data.get('predictor').get('resource').get('limits').get('memory')
        resource_limits_gpu = request_data.get('predictor').get('resource').get('limits').get('gpu')
        resource_requests_cpu = request_data.get('predictor').get('resource').get('limits').get('cpu')
        resource_requests_memory = request_data.get('predictor').get('resource').get('limits').get('memory')
  
        predictor = Predictor(model_framwwork, 
                              runtime_version, 
                              storage_uri, 
                              logger, 
                              resource_limits_cpu, 
                              resource_limits_memory, 
                              resource_limits_gpu, 
                              resource_requests_cpu, 
                              resource_requests_memory)
    
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

    storage_credential = request_data.get('storage_credential')
    storage_endpoint = request_data.get('storage_endpoint')    
    
    model_storage = ModelStorage(storage_credential, 
                                 storage_endpoint)

    if(storage_credential == None and storage_endpoint == None):
        model_storage.storage_credential = os.environ['KSERVE_API_DEFAULT_STORAGE_CREDENTIAL']
        model_storage.storage_endpoint = os.environ['KSERVE_API_DEFAULT_STORAGE_ENDPOINT']
    
    return predictor, transformer, model_storage


def make_credential(model_storage, predictor):
    #make credential & spec string
    print(model_storage.storage_credential)
    if(predictor.storage_uri.find("s3:") >= 0):
        with open('/tmp/credential.txt', 'w') as credential_file:
            credential_file.write(model_storage.storage_credential)        
        KServe.set_credentials(storage_type='s3', namespace=target_namespace, credentials_file=credential_file.name, s3_profile='default', s3_endpoint=model_storage.storage_endpoint, s3_region='us-west-2', s3_use_https='1', s3_verify_ssl='0') 
        return True
    else:
        return False


def make_spec(predictor, inference_name, transformer):
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
    if(predictor.runtime_version == None):
        runtime_verion_spec_str = ", runtime_version=" + predictor.runtime_version 
    else:
        runtime_verion_spec_str = ""

    #make logger spec string
    if(predictor.logger != None):
        predictor_logger_spec_str = ", logger=V1beta1LoggerSpec(mode=\'" + predictor.logger + "\',url=\'" + os.environ['KSERVE_API_LOGGERURL'] + "\')"
    else:
        predictor_logger_spec_str = ""

    predictor_spec_str = "predictor=V1beta1PredictorSpec(" + \
                                                          model_framwwork_spec_str + \
                                                          "(storage_uri='" + predictor.storage_uri + "'" + \
                                                             predictor_resource_spec_str + \
                                                             runtime_verion_spec_str + \
                                                           ")"  + \
                                                          predictor_logger_spec_str +")"


    #make transformer spec string
    if(transformer.image != None):
        if(transformer.logger != None):
            transformer_logger_spec_str = ", logger=V1beta1LoggerSpec(mode=\'" + transformer.logger + "\',url=\'" + os.environ['KSERVE_API_LOGGERURL'] + "\')"
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

        transformer_spec_str = ", transformer=V1beta1TransformerSpec(" \
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
    #동시요청수가 10을 넘으면 확장
    autoscaling_spec_str = "{\'autoscaling.knative.dev/target\': \'10\'}"
     
    #make iference service spec
    isvc = V1beta1InferenceService(api_version=constants.KSERVE_V1BETA1,
                                  kind=constants.KSERVE_KIND,
                                  metadata=client.V1ObjectMeta(name=inference_name, namespace=target_namespace, annotations=eval(autoscaling_spec_str)),
                                  spec=eval(default_model_spec))
    print(isvc)
    return isvc     



############################################################
# Main
############################################################
if __name__ == '__main__':
    app.run()
    logger.info("Kserve-API Server Start!!")    