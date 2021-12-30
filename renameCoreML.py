#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""
import coremltools as ct

def main():
    # get model specification
    model_path = "./pretrain_model/exp-schp-atr.mlmodel"
    mlmodel = ct.models.MLModel(str(model_path))
    spec = mlmodel.get_spec()

    # get list of current output_names
    current_output_names = mlmodel.output_description._fd_spec

    # rename first output in list to new_output_name
    old_name = current_output_names[0].name
    new_name = "output"
    ct.utils.rename_feature(
        spec, old_name, new_name, rename_outputs=True
    )

    # overwite existing model spec with new renamed spec
    new_model = ct.models.MLModel(spec)
    new_model.save(model_path)

if __name__ == '__main__':
    main()
