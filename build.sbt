name := "AnalyticsZooScala"

version := "0.1"

scalaVersion := "2.11.8"


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.4.5",
  "org.apache.spark" %% "spark-sql" % "2.4.5",
  "org.scalatest" %% "scalatest" % "3.0.8" % "test",
  "org.apache.spark" %% "spark-mllib" % "2.4.5",
// figure out % and %% for dependencies
  "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.10.0-spark_2.4.3" % "0.8.1"
)
