Installation
============

Requirements
------------

* Python 3.9+
* ``numpy>=1.22``

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/chenxinye/noisefloat2.git
   cd noisefloat2
   pip install -e .

Optional backend extras
-----------------------

.. code-block:: bash

   pip install torch        # PyTorch backend
   pip install jax jaxlib   # JAX backend
   pip install tensorflow   # TensorFlow backend
   pip install scipy        # Improved Student-t values

Documentation build dependencies
--------------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt

Build docs locally
------------------

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in your browser.
