﻿using System;
using System.IO;
using System.Linq;
using Common;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;
using SentimentAnalysis.DataStructures;

namespace SentimentAnalysis
{
    internal static class Program
    {
        private static readonly string BaseDatasetsRelativePath = @"Data";
        private static readonly string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/wikipedia-detox-250-line-data.tsv";
        private static readonly string TestDataRelativePath = $"{BaseDatasetsRelativePath}/wikipedia-detox-250-line-test.tsv";
        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static readonly string BaseModelsRelativePath = @"../../../MLModels";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/SentimentModel.zip";
        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        private static uint ExperimentTime = 60;

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Create, train, evaluate and save a model
            BuildTrainEvaluateAndSaveModel(mlContext);
            ConsoleHelper.ConsoleWriteHeader("=============== End of training process ===============");

            // Make a single test prediction loading the model from .ZIP file
            TestSinglePrediction(mlContext);

            ConsoleHelper.ConsoleWriteHeader("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            // STEP 1: Load data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(TrainDataPath, hasHeader: true);
            IDataView testDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(TestDataPath, hasHeader: true);

            // STEP 2: Display first few rows of training data
            ConsoleHelper.ShowDataViewInConsole(mlContext, trainingDataView);

            // STEP 3: Initialize our user-defined progress handler that AutoML will 
            // invoke after each model it produces and evaluates.
            var progressHandler = new BinaryExperimentProgressHandler();

            // STEP 4: Run AutoML binary classification experiment
            Console.WriteLine($"Running AutoML binary classification experiment for {ExperimentTime} seconds...");
            ExperimentResult<BinaryClassificationMetrics> experimentResult = mlContext.Auto()
                .CreateBinaryClassificationExperiment(ExperimentTime)
                .Execute(trainingDataView, progressHandler: progressHandler);

            // STEP 5: Evaluate the model and print metrics
            Console.WriteLine("===== Evaluating model's accuracy with test data =====");
            RunDetail<BinaryClassificationMetrics> bestRun = experimentResult.BestRun;
            ITransformer trainedModel = bestRun.Model;
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(data:predictions, scoreColumnName: "Score");
            ConsoleHelper.PrintBinaryClassificationMetrics(bestRun.TrainerName, metrics);

            // Print top models found by AutoML
            PrintTopModels(experimentResult);

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            return trainedModel;
        }

        // (OPTIONAL) Try/test a single prediction by loading the model from the file, first.
        private static void TestSinglePrediction(MLContext mlContext)
        {         
            SentimentIssue sampleStatement = new SentimentIssue { Text = "This is a very rude movie" };
            
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);       
            Console.WriteLine($"=============== Loaded Model OK  ===============");               

            // Create prediction engine related to the loaded trained model
            var predEngine= mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);
            Console.WriteLine($"=============== Created Prediction Engine OK  ==============="); 
            // Score
            var predictedResult = predEngine.Predict(sampleStatement);

            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(predictedResult.Prediction) ? "Toxic" : "Non Toxic")} sentiment");
            Console.WriteLine($"==================================================");
        }
        
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        /// <summary>
        /// Prints top models from AutoML experiment.
        /// </summary>
        private static void PrintTopModels(ExperimentResult<BinaryClassificationMetrics> experimentResult)
        {
            // Get top few runs ranked by accuracy
            var topRuns = experimentResult.RunDetails.OrderByDescending(r => r.ValidationMetrics.Accuracy).Take(3);

            Console.WriteLine($"Top models ranked by accuracy --");
            ConsoleHelper.PrintBinaryClassificationMetricsHeader();
            for (var i = 0; i < topRuns.Count(); i++)
            {
                var run = topRuns.ElementAt(i);
                ConsoleHelper.PrintIterationMetrics(i + 1, run.TrainerName, run.ValidationMetrics, run.RuntimeInSeconds);
            }
        }
    }
}
