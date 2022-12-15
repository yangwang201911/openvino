.. OpenVINO Toolkit documentation master file, created by
   sphinx-quickstart on Wed Jul  7 10:46:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :google-site-verification: _YqumYQ98cmXUTwtzM_0WIIadtDc6r_TMYGbmGgNvrk

.. rst-class:: openvino-intro-text

   OpenVINO is an open-source toolkit for optimizing and deploying deep learning models. It provides boosted deep learning performance for vision, audio, and language models from popular frameworks like TensorFlow, PyTorch, and more. `Get started with OpenVINO. <get_started.html>`__

.. rst-class:: openvino-chart

   .. image:: _static/images/openvino_diagram.svg


Overview
~~~~~~~~

OpenVINO enables you to optimize a deep learning model from almost any framework and deploy it with best-in-class performance on a range of Intel  processors and other hardware platforms.

A typical workflow with OpenVINO is shown below.

.. raw:: html

   <div class="section" id="welcome-to-openvino-toolkit-s-documentation">

   <link rel="stylesheet" type="text/css" href="_static/css/homepage_style.css">

      <div style="clear:both;"> </div>

      <div id="HP_flow-container">
		   <div class="HP_flow-btn">
	   		<a href="https://docs.openvino.ai/latest/openvino_docs_model_processing_introduction.html">
	   			<img src="_static/images/OV_flow_model_hvr.svg" alt="link to model processing introduction" /> 
	   		</a>
	   	</div>
	   	<div class="HP_flow-arrow" >
	   			<img src="_static/images/OV_flow_arrow.svg" alt="" /> 
	   	</div>
	   	<div class="HP_flow-btn">
	   		<a href="https://docs.openvino.ai/latest/openvino_docs_deployment_optimization_guide_dldt_optimization_guide.html">
	   			<img src="_static/images/OV_flow_optimization_hvr.svg" alt="link to an optimization guide" /> 
	   		</a>
	   	</div>
	   	<div class="HP_flow-arrow" >
	   			<img src="_static/images/OV_flow_arrow.svg" alt="" /> 
	   	</div>
	   	<div class="HP_flow-btn">
	   		<a href="https://docs.openvino.ai/latest/openvino_docs_deployment_guide_introduction.html">
	   			<img src="_static/images/OV_flow_deployment_hvr.svg" alt="link to deployment introduction" /> 
	   		</a>
		   </div>
	   </div>


High-Performance Deep Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenVINO Runtime automatically optimizes deep learning pipelines using aggressive graph fusion, memory reuse, load balancing, and inferencing parallelism across CPU, GPU, VPU, and more.
You can integrate and offload to accelerators additional operations for pre- and post-processing to reduce end-to-end latency and improve throughput.

Model Quantization and Compression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Boost your model’s speed even further with quantization and other state-of-the-art compression techniques available in OpenVINO’s Post-Training Optimization Tool and Neural Network Compression Framework. These techniques also reduce your model size and memory requirements, allowing it to be deployed on resource-constrained edge hardware. 

.. panels::
   :card: homepage-panels

   **Local Inferencing & Model Serving**

   You can either link directly with OpenVINO Runtime to run inference locally or use OpenVINO Model Serving to serve model inference from separate server or within Kubernetes environment

   ---

   **Improved Application Portability**

   Write an application once, deploy it anywhere, achieving maximum performance from hardware. Automatic device discovery allows for superior deployment flexibility. OpenVINO Runtime supports Linux, Windows and MacOS and provides Python, C++ and C API. Use your preferred language and OS.

   ---

   **Minimal External Dependencies**

   Designed with minimal external dependencies reduces the application footprint, simplifying installation and dependency management. Popular package managers enable application dependencies to be easily installed and upgraded. Custom compilation for your specific model(s) further reduces final binary size.

   ---

   **Enhanced App Start-Up Time**

   In applications where fast start-up is required, OpenVINO significantly reduces first-inference latency by using the CPU for initial inference and then switching to GPU or VPU once the model has been compiled and loaded to memory. Compiled models are cached to further improving start-up time.


Supported Devices
~~~~~~~~~~~~~~~~~

OpenVINO is supported on a wide range of hardware platforms. Visit the `Supported Devices <openvino_docs_OV_UG_supported_plugins_Supported_Devices.html>`__ page for a full list of OpenVINO-compatible platforms.

* All Intel Xeon, Core, and Atom CPUs, with boosted performance on 11th generation Core CPUs and 3rd generation Xeon CPUs or newer
* Intel integrated GPUs including Intel UHD Graphics and Intel Iris Xe
* Intel discrete GPUs including Iris Xe MAX and Arc
* Intel accelerators such as VPUs and GNAs
* Arm CPU, including Apple ARM based SoCs

Check the `Performance Benchmarks <openvino_docs_performance_benchmarks.html>`__ page to see how fast OpenVINO runs popular models on a variety of processors. OpenVINO supports deployment on Windows, Linux, and macOS.

Install OpenVINO
~~~~~~~~~~~~~~~~

`Go to installation to set up OpenVINO on your device. <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html>`__

Get Started
~~~~~~~~~~~

`Visit the Get Started Guide to learn the basics of OpenVINO and explore its features with quick start examples. <get_started.html>`__


.. toctree::
   :maxdepth: 2
   :hidden:

   GET STARTED <get_started>
   LEARN OPENVINO <learn_openvino>
   OPENVINO WORKFLOW <openvino_workflow>
   DOCUMENTATION <documentation>
   MODEL ZOO <model_zoo>
   RESOURCES <resources>
   RELEASE NOTES <https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html>

