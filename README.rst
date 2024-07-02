openlifu
========

|Actions Status| |Documentation Status|

|PyPI version| |PyPI platforms|

|GitHub Discussion|

.. SPHINX-START

``openlifu`` is a toolbox for planning and controlling focused
ultrasound treatments. It generally replicates the functionality
developed in the MATLAB `open-TFUS
toolbox <https://github.com/OpenwaterHealth/opw_neuromod_sw>`__.

Installation
------------

Requirements
~~~~~~~~~~~~

Python 3.9 or later on Windows or Linux.

Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Windows:

.. code:: sh

   C:\Users\<uname>\AppData\Local\Programs\Python\Python311\python.exe -m venv env
   .\env\Scripts\activate

Linux:

.. code:: sh

   python3.11 -m venv env

Install project (editable)
~~~~~~~~~~~~~~~~~~~~~~~~~~

With this repo as the working directory:

.. code:: sh

   pip install -e .

Disclaimer
----------

CAUTION - Investigational device. Limited by Federal (or United States)
law to investigational use. The system described here has *not* been
evaluated by the FDA and is not designed for the treatment or diagnosis
of any disease. It is provided AS-IS, with no warranties. User assumes
all liability and responsibility for identifying and mitigating risks
associated with using this software.

.. |Actions Status| image:: https://github.com/OpenwaterHealth/OpenLIFU-python/workflows/CI/badge.svg
   :target: https://github.com/OpenwaterHealth/OpenLIFU-python/actions
.. |Documentation Status| image:: https://readthedocs.org/projects/openlifu/badge/?version=latest
   :target: https://openlifu.readthedocs.io/en/latest/?badge=latest
.. |PyPI version| image:: https://img.shields.io/pypi/v/openlifu
   :target: https://pypi.org/project/openlifu/
.. |PyPI platforms| image:: https://img.shields.io/pypi/pyversions/openlifu
   :target: https://pypi.org/project/openlifu/
.. |GitHub Discussion| image:: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
   :target: https://github.com/OpenwaterHealth/OpenLIFU-python/discussions
