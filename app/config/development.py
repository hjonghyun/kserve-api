from config.default import *

DEFAULT_STORAGE_ENDPOINT = "192.168.88.154:9000"     #모델이 저장된 storage 정보 기본값으로 내부 스토리지(minio) 설정됨 
DEFAULT_STORAGE_CREDENTIAL = "[default] \n AWS_ACCESS_KEY_ID:bWluaW9hZG1pbg== \n AWS_SECRET_ACCESS_KEY:bWluaW9hZG1pbg=="  #모델이 저장된 storage 인증정보 
LOGGERURL = "http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker"  #knative Request/Response 정보 이벤트를 전달할 서비스위치   
K8S_CONFIG = "/.kube/config"      #inference service 배포할 K8S Cluster의 config 위치 