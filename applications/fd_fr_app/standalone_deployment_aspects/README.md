* This folder contains the deployment code required for standalone testing of individual IoT microservices. This will be used for normal data generation (including microservice property 
characterization, saturation point identification).
* Following is the order of deployment

* Setup multi-node cluster in k3s s.t. there is a node for standalone_k8s_master (8c32g), standalone_worker (choose based on requirement), standalone_jmeter (2c8g)

  * Apply the service and deployment yamls of the target microservice. If you are testing, 
    * preprocessor -
      * `kubectl apply -f preprocessor_deployment.yaml`
      * `kubectl apply -f preprocessor_service.yaml`
    * fd
      * `kubectl apply -f fd_deployment.yaml`
      * `kubectl apply -f fd_service.yaml`
    * fr  
      * `kubectl apply -f fr_deployment.yaml`
      * `kubectl apply -f fr_service.yaml`

* Installing jmeter and sending a workload - 
  * Install jmeter in your k8s cluster - https://dev.to/hitjethva/how-to-install-apache-jmeter-on-ubuntu-20-04-2di9 
  * Locally test and copy the jmx file to the k8s cluster
  *  `./jmeter -n -t path/to/testplan.jmx -l path/to/results.jtl`

* Deploy Pixie in the master node
