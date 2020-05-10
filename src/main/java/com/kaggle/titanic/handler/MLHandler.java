package com.kaggle.titanic.handler;

import com.kaggle.titanic.data.machine.learning.Storage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MLHandler {
    final WeightInit weightInit;
    final Activation activation;

    private final Storage storage;
    MultiLayerConfiguration conf;
    MultiLayerNetwork model;
    Future<Boolean> future;
    private ExecutorService executorService;

    public MLHandler(
        Storage storage,
        int seed,
        int iterations,
        Double weightDecay,
        Double weightDecayBias,
        int numInputs,
        int outputNum
    ) {
        this.storage = storage;

        activation = Activation.GELU;
        weightInit = WeightInit.XAVIER;

        conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .maxNumLineSearchIterations(iterations)
            .activation(activation)
            .weightInit(weightInit)
            .weightDecay(weightDecay)
            .weightDecayBias(weightDecayBias)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
            .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
            .layer(
                2,
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build()
            )
            .build();
    }

    public void initModel(int printIterations) throws IOException {
        if (getTempFile().exists() && !getTempFile().isDirectory()) {
            model = MultiLayerNetwork.load(getTempFile(), false);
            model.setLayerWiseConfigurations(conf);
            model.setListeners(new ScoreIterationListener(printIterations));

            return;
        }

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(printIterations));
    }

    public void learn(int iterations) {
        executorService = Executors.newSingleThreadExecutor();

        future = executorService.submit(() -> {
            for (int i = 0; i < iterations; i++) {
                model.fit(storage.getTrainData());
            }

            return true;
        });
    }

    public void stopThread() {
        executorService.shutdown();
    }

    public Future<Boolean> getFuture() {
        return future;
    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public void saveModel() throws IOException {
        this.getModel().save(getTempFile());
    }

    private File getTempFile() {
        String folder = "models";
        File directory = new File(folder);

        if (! directory.exists()) {
            //noinspection ResultOfMethodCallIgnored
            directory.mkdir();
        }

        String fileAdd = folder + "/" + activation.getDeclaringClass().getCanonicalName() + "_" + weightInit.getDeclaringClass().getCanonicalName();

        return new File(fileAdd + "_train.dataModel");
    }
}
