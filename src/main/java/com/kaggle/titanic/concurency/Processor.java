package com.kaggle.titanic.concurency;

import com.kaggle.titanic.data.machine.learning.Storage;
import com.kaggle.titanic.handler.MLHandler;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Processor {
    List<MLHandler> MLHandlerList = new ArrayList<>();
    Storage storage;

    public Processor(Storage storage) {
        this.storage = storage;
    }

    public void initProcessEval(int iterations, double weightDecay, double weightDecayBias, int seed) throws IOException {
        final int numInputs = 11;
        int outputNum = 2;

        System.out.println("Build model....");
        MLHandler mlHandler = new MLHandler(
            storage,
            seed,
            iterations,
            weightDecay,
            weightDecayBias,
            numInputs,
            outputNum
        );
        MLHandlerList.add(mlHandler);

        mlHandler.initModel(100);
    }

    public void initProcess(int iterations, double weightDecay, double weightDecayBias, int seed) throws IOException {
        //Configure a simple model. We're not using an optimal configuration here, in order to show evaluation/errors, later
        storage.normalizeData();
        final int numInputs = 11;
        int outputNum = 2;

        System.out.println("Build model....");
        MLHandler mlHandler = new MLHandler(
            storage,
            seed,
            iterations,
            weightDecay,
            weightDecayBias,
            numInputs,
            outputNum
        );
        MLHandlerList.add(mlHandler);

        mlHandler.initModel(100);
    }

    public void learn(int iterations) {
        for (MLHandler handler : MLHandlerList) {
            handler.learn(iterations);
        }

//        boolean learningInProgress = true;
//        while(learningInProgress)
//        {
//            Thread.sleep(1000);
//            learningInProgress = false;
//
//            for (MLHandler handler: MLHandlerList) {
//                if (
//                    ! handler.getFuture().isCancelled() &&
//                    ! handler.getFuture().isDone()
//                ) {
//                    learningInProgress = true;
//                    break;
//                }
//            }
//        }

        for (MLHandler handler : MLHandlerList) {
            try {
                handler.getFuture().get();
                handler.saveModel();
            } catch (Exception ignored) {

            } finally {
                handler.stopThread();

            }
        }
    }

    public MultiLayerNetwork getModel() {
        return this.MLHandlerList.get(0).getModel();
    }
}
