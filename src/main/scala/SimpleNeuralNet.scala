import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models._
//import  com.intel.analytics.zoo.pipeline.api.
import com.intel.analytics.bigdl.utils.Shape

object SimpleNeuralNet {
  def main(args: Array[String]): Unit = {
    println("TRAINING SIMPLE NEURAL NETWORK!")

    val spark = SparkSession
      .builder
      .appName("Spark_Basic_Learning")
      .master("local[2]") //SANG: FOR LOCAL testing
      .config("spark.sql.warehouse.dir", "file:///C:/Spark/temp")
      .config("spark.sql.streaming.checkpointLocation", "file:///C:/Spark/checkpoint")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val df = spark.read.format("csv")
      .option("mode", "FAILFAST")
      .option("inferSchema", "true")
      .option("path", "./src/resources/dataset-1_converted.csv")
      .option("header", "true")
      .load()

    df.show()

    println("Imported libs work!")

    val Array(trainDf, testDf) = df.randomSplit(Array(0.8, 0.2))
    /*
     x:Input columns
     y:Output columns
    */
    val inputs = 2
    val outputs = 1

    val model = Sequential()

    model.add( Dense( outputDim = inputs,activation = "relu",inputShape =Shape(inputs)) )
    model.add( Dense(outputDim = outputs,activation = "relu") )


    println("Simple Multiperceptron Created")
    model.summary()



  }


}
