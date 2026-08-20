Maintenance handoff
===================

Primary maintainer
------------------

Current maintenance owner:

* **Xinye Chen**
* Website: https://xinyechen.com
* Repository: https://github.com/chenxinye/noisefloat2

Operations checklist
--------------------

For each release cycle:

1. Run unit tests:

   .. code-block:: bash

      pytest -v

2. Build documentation:

   .. code-block:: bash

      pip install -r docs/requirements.txt
      sphinx-build -b html docs docs/_build/html

3. Verify API docs render correctly under ``docs/api/``.
4. Review instability example scripts in ``examples/`` if numeric behavior changed.

Read the Docs setup
-------------------

This repository includes ``/home/runner/work/noisefloat2/noisefloat2/.readthedocs.yaml``.
Read the Docs will:

* install the project package
* install doc dependencies from ``docs/requirements.txt``
* build from ``docs/conf.py`` with warnings treated as errors

Documentation design note
-------------------------

The Furo theme is configured for a clean, restrained visual style with a
minimal color system and Japanese-oriented sans-serif font stack to keep pages
calm, modern, and easy to scan.
