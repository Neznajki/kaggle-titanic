package com.kaggle.titanic.data.machine.learning;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Transformer {
    public void transformDataForMachineLearning(String passengerFile, String targetFile) throws InterruptedException, IOException {
        Schema.Builder builder = new Schema.Builder();

        //PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        //Age,SibSp,Parch,Ticket
        //Fare
        builder
            .addColumnInteger("PassengerId")
        ;

        if (targetFile.matches(".*train.*")) {
            builder.addColumnBoolean("Survived");
        }

        builder
            .addColumnInteger("Pclass")
            .addColumnsString("Name")
            .addColumnCategorical("Sex", "", "male", "female")
            .addColumnsString("Age")
            .addColumnInteger("SibSp")
            .addColumnInteger("Parch")
            .addColumnsString("Ticket")
            .addColumnFloat("Fare")
            .addColumnsString("Cabin")
            .addColumnCategorical("Embarked", "", "S", "C", "Q")
        ;

        Map<String, String> cabinMapping = new HashMap<>();

        cabinMapping.put("^(.*)A(.*)$", "100$2");
        cabinMapping.put("^(.*)B(.*)$", "200$2");
        cabinMapping.put("^(.*)C(.*)$", "300$2");
        cabinMapping.put("^(.*)D(.*)$", "400$2");
        cabinMapping.put("^(.*)E(.*)$", "500$2");
        cabinMapping.put("^(.*)F(.*)$", "600$2");
        cabinMapping.put("^(.*)F E(.*)$", "700$2");
        cabinMapping.put("^(.*)G(.*)$", "800$2");
        cabinMapping.put("^(.*)T(.*)$", "900$2");
        cabinMapping.put("^$", "0");

        Map<String, String> ticketMapping = new HashMap<>();

        ticketMapping.put("^.*([0-9]+)$", "$1");
        ticketMapping.put("^LINE$", "666");
        ticketMapping.put("^$", "0");

        Map<String, String> nameMapping = new HashMap<>();

        nameMapping.put("^.*Mrs\\..*$", "1");
        nameMapping.put("^.*Mr\\..*$", "2");
        nameMapping.put("^.*Miss\\..*$", "3");
        nameMapping.put("^.*Ms\\..*$", "4");
        nameMapping.put("^.*Master\\..*$", "5");
        nameMapping.put("^.*$", "0");

        Map<String, String> ageMapping = new HashMap<>();

        ageMapping.put("^\\s*([0-9]+)\\s*$", "$1");
        ageMapping.put("^\\s*$", "0");

        TransformProcess tp = new TransformProcess.Builder(builder.build())
            .replaceStringTransform("Name", nameMapping)
            .replaceStringTransform("Age", ageMapping)
            .replaceStringTransform("Ticket", ticketMapping)
            .replaceStringTransform("Cabin", cabinMapping)
            .categoricalToInteger(
                "Sex",
                "Embarked"
            ).build();


        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Example");

        JavaSparkContext sc = new JavaSparkContext(conf);

        RecordReader rr = new CSVRecordReader();

        JavaRDD<String> fileContents = sc
            .textFile(passengerFile);

        String header = fileContents.first();

        JavaRDD<List<Writable>> passengerInfo = fileContents
            .filter((String line) -> !line.equals(header))
            .map(new StringToWritablesFunction(rr));

        JavaRDD<List<Writable>> passengerData =
            SparkTransformExecutor.execute(
                passengerInfo,
                tp
            );

        JavaRDD<String> processedAsString = passengerData.map(new WritablesToStringFunction(","));
        List<String> processedCollected = processedAsString.collect();
//        List<List<Writable>> joinedDataList = joinedData.collect();

        //Stop spark, and wait a second for it to stop logging to console
        sc.stop();
        Thread.sleep(500);

        //Print the original data
//        System.out.println("\n\n----- Customer Information -----");
//        System.out.println("Source file: " + loanSourceDataFile);
//        System.out.println(loanData);
//        System.out.println("Customer Information Data:");
//        for(List<Writable> line : loanInfoList){
//            System.out.println(line);
//        }


//        System.out.println("\n\n----- Purchase Information -----");
//        System.out.println("Source file: " + HistorySourceDataFile);
//        System.out.println(loanHistoryData);
//        System.out.println("Purchase Information Data:");
//        for(List<Writable> line : historyInfoList){
//            System.out.println(line);
//        }

//        System.out.println("Joined Data:");
//        for(String line : processedCollected){
//            System.out.println(line);
//        }
//        System.out.println(processedCollected);
//        System.out.println("\n\n----- Joined Data -----");
//        System.out.println(tp.getFinalSchema());
//
        Path out = Paths.get(targetFile);
        Files.write(out, processedCollected, Charset.defaultCharset());
    }
}
