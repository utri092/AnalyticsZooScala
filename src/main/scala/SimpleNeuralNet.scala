import com.intel.analytics.bigdl.nn.tf
import org.apache.spark.{SparkConf, SparkContext, sql}
import org.apache.spark.sql.{Row, SparkSession}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{ConvertModel, Engine}
import com.intel.analytics.zoo.models.image.imageclassification.Dataset
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models._
import com.intel.analytics.zoo.pipeline.api._
import com.intel.analytics.bigdl.dlframes.DLModel
import com.intel.analytics.zoo.pipeline.api.keras.python.PythonZooKeras2
import com.intel.analytics.zoo.pipeline.nnframes.{NNEstimator, NNModel}
import org.apache.spark.sql.types.{IntegerType, LongType}
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.zoo.tfpark.{TFDataFeatureSet, TFMiniBatch}
import org.apache.spark.rdd.RDD

object SimpleNeuralNet {
  def main(args: Array[String]): Unit = {
    println("TRAINING SIMPLE NEURAL NETWORK!")

    val conf = Engine.createSparkConf()
      .setAppName("Spark_Basic_Learning")
      .setMaster("local[2]") //SANG: FOR LOCAL testing
      .set("spark.sql.warehouse.dir", "file:///C:/Spark/temp")
      .set("spark.sql.streaming.checkpointLocation", "file:///C:/Spark/checkpoint")
      .set("spark.testing.memory", "471859200")
//      .set("spark.sql.execution.arrow.pyspark.enabled", "true")

    val sc = new SparkContext(conf)

    val spark = SparkSession.builder().config(conf).getOrCreate()

    //Init BigDL Engine
    Engine.init

    val df = spark.read.format("csv")
      .option("mode", "FAILFAST")
      .option("inferSchema", "true")
      .option("path", "./src/resources/dataset-1_converted.csv")
      .option("header", "true")
      .load()

//    val rows: RDD[Row] = df.rdd

//    println(rows, rows.getClass())

//    df.show()

//    val rdd = spark.sparkContext.textFile("./src/resources/dataset-1_converted.csv")
   /* rdd.foreach(f=>{
      println(f)
    })*/

//    println(rdd, rdd.getClass())

    println("Imported libs work!")

    val Array(trainDf, testDf) = df.randomSplit(Array(0.8, 0.2))

//    val Array(trainRdd, testRdd) = rdd.randomSplit(Array(0.8, 0.2))

  /* x:Input columns
     y:Output columns*/
    val inputs = 2
    val outputs = 1

    var kModel = Sequential()

    kModel.add(Dense( outputDim = inputs,activation = "relu",inputShape = Shape(inputs)) )
    kModel.add(Dense(outputDim = outputs,activation = "relu") )
    kModel.compile(loss = "mse", optimizer = "adam")
//    val model2 = kModel.toModel()
//    model2.saveToTf("./src/resources/trialModel.pb", python="./src/pyVenv/bin/python3")
//    kModel.summary()
//    kModel.saveToTf("./src/resources/trialModel.pb", python="./src/pyVenv/bin/python3")
//    kModel.saveToKeras2("./src/resources/trialKerasModel.h5", python = "./src/pyVenv/bin/python3")
//    kModel.saveModule("./src/resources/trialModel.pb", overWrite = true)

    println("Simple Multiperceptron Created")

//    kModel.saveTorch("./src/resources/lol.pt", true)

//    var trainSet = DataSet.array(Array(trainDf.select("carparkID"), trainDf.select("processing-time"), trainDf.select("slotOccupancy")), sc)

    kModel.saveWeights("./src/resources/kerasWeights.txt", true)
//    kModel.saveDefinition("./src/resources/kerasDef.json", true)
//    kModel.saveToTf("./src/resources/tfModel.pb")
    val nnModel = NNModel(kModel)
//    nnModel.model.predict(da)
//    nnModel.model.saveWeights("./src/resources/SavedModels/keras/weights/weights.h5", true)
//    nnModel.model.save("./src/resources/SavedModels/keras/model.h5")
//    nnModel.model.save("./src/resources/SavedModels/tf/model.pb")
//    nnModel.model.saveModule("./src/resources/SavedModels/tf/model2.pb")
//    nnModel.model.saveWeights("./src/resources/SavedModels/tf/model2.pb)
//    nnModel.model.saveDefinition("./src/resources/SavedModels/keras/model2.json", true)
//    nnModel.save("./src/resources/nnh5Model.h5")
//    nnModel.model.saveWeights("./src/resources/model_weights.h5", true)
//    nnModel.model.saveDefinition("./src/resources/model_def.json", overWrite = true)
//    nnModel.model.saveModule("./src/resources/nnTrialModel.pb", overWrite = true)
//    kModel.summary()
    // Does not work !
//    kModel.fit(trainDf[0],10,32)
// Does not work !
//      kModel.fit(trainRdd)


  }


}
