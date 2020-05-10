package com.kaggle.titanic.entry;

import com.kaggle.titanic.data.machine.learning.Transformer;
import com.kaggle.titanic.helper.Resources;

public class DataTransformer {

    public static void main(String[] args) throws Exception {
        Transformer dataTransformer = new Transformer();

        dataTransformer.transformDataForMachineLearning(
            (new Resources("train.csv")).getAbsolutePath(),
            "train_file.csv"
        );

        dataTransformer.transformDataForMachineLearning(
            (new Resources("test.csv")).getAbsolutePath(),
            "test_file.csv"
        );
    }
}
