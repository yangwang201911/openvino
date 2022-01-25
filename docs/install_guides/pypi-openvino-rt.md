# OpenVINO™ Runtime

## Introduction

OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. Based on latest generations of artificial neural networks, including Convolutional Neural Networks (CNNs), recurrent and attention-based networks, the toolkit extends computer vision and non-vision workloads across Intel® hardware, maximizing performance. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud.

[OpenVINO™ Runtime](https://docs.openvino.ai/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html) package for Python includes a set of libraries for an easy inference integration into your Python applications and   supports of heterogeneous execution across Intel® CPU and Intel® GPU hardware.

## System Requirements
The complete list of supported hardware is available in the [Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino-relnotes.html).

The table below lists supported operating systems and Python* versions required to run the installation.

| Supported Operating System                                   | [Python* Version (64-bit)](https://www.python.org/) |
| :------------------------------------------------------------| :---------------------------------------------------|
|   Ubuntu* 18.04 long-term support (LTS), 64-bit              | 3.6, 3.7, 3.8                                       |
|   Ubuntu* 20.04 long-term support (LTS), 64-bit              | 3.6, 3.7, 3.8, 3.9                                  |
|   Red Hat* Enterprise Linux* 8, 64-bit                       | 3.6, 3.8                                            |
|   macOS* 10.15.x versions                                    | 3.6, 3.7, 3.8, 3.9                                  |
|   Windows 10*, 64-bit                                        | 3.6, 3.7, 3.8, 3.9                                  |

> **NOTE**: This package can be installed on other versions of Linux and Windows OSes, but only the specific versions above are fully validated.

> **NOTE**: The current version of the OpenVINO™ Runtime for macOS* supports inference on Intel® CPUs only.

## Install the OpenVINO™ Runtime Package

### Step 1. Set Up Python Virtual Environment

To avoid dependency conflicts, use a virtual environment. Skip this
   step only if you do want to install all dependencies globally.

Create virtual environment:
```sh
python -m pip install --user virtualenv 
python -m venv openvino_env
```

> **NOTE**: On Linux and macOS, you may need to type `python3` instead of
`python`. You may also need to [install pip](https://pip.pypa.io/en/stable/installing/). For example, on Ubuntu execute the following command to get pip installed: `sudo apt install python3-venv python3-pip`.

### Step 2. Activate Virtual Environment

On Linux and macOS:
```sh
source openvino_env/bin/activate
```
On Windows:
```sh
openvino_env\Scripts\activate
```

### Step 3. Set Up and Update PIP to the Highest Version

Run the command below:
```sh
python -m pip install --upgrade pip
```

### Step 4. Install the Package

Run the command below: <br>

   ```sh
   pip install openvino
   ```

### Step 5. Verify that the Package Is Installed

Run the command below:
```sh
python -c "from openvino.runtime import Core"
```
   
If installation was successful, you will not see any error messages (no console output).

## Troubleshooting

### Error: Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio"

On Windows* some dependencies may require compilation from source when installing. To resolve this issue, you need to install [Build Tools for Visual Studio* 2019](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019) and repeat package installation.

### ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory

To resolve missing external dependency on Ubuntu*, execute the following command:
```sh
sudo apt-get install libpython3.7
```

## Additional Resources

- [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit)
- [OpenVINO™ toolkit online documentation](https://docs.openvino.ai)
- [OpenVINO™ Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

Copyright © 2018-2022 Intel Corporation
> **LEGAL NOTICE**: Your use of this software and any required dependent software (the
“Software Package”) is subject to the terms and conditions of the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html) for the Software Package, which may also include notices, disclaimers, or
license terms for third party or open source software included in or with the Software Package, and your use indicates your acceptance of all such terms. Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package for additional details.

>Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the [Intel Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right.
