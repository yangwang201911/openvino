.. {#openvino_docs_OV_UG_lpt_step1_prerequisites}

Step 1. Prerequisites Transformations
=====================================


.. meta::
   :description: Learn about optional Prerequisites transformations, that
                 prepare a model before applying other low precision transformations.

.. toctree::
   :maxdepth: 1
   :hidden:

   PullReshapeThroughDequantization <step1-prerequisites/pull-reshape-through-dequantization>
   PullTransposeThroughDequantization <step1-prerequisites/pull-transpose-through-dequantization>
   LinOpSequenceFusion <step1-prerequisites/lin-op-sequence-fusion>

Prerequisites transformations are optional. The transformations prepare a model before running other low precision transformations. The transformations do not operate with dequantization operations or update precisions. Prerequisites transformations include:

* :doc:`PullReshapeThroughDequantization <step1-prerequisites/lin-op-sequence-fusion>`
* :doc:`PullTransposeThroughDequantization <step1-prerequisites/pull-transpose-through-dequantization>`
* :doc:`LinOpSequenceFusion <step1-prerequisites/lin-op-sequence-fusion>`

