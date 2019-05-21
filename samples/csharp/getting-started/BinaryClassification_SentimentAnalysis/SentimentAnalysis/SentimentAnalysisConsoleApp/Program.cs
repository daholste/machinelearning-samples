using System;
using System.IO;
using Microsoft.ML;
using SentimentAnalysisConsoleApp.DataStructures;
using Common;
using static Microsoft.ML.DataOperationsCatalog;
using System.Collections.Generic;

namespace SentimentAnalysisConsoleApp
{
    internal static class Program
    {
        private static readonly string BaseDatasetsRelativePath = @"../../../../Data";
        private static readonly string DataRelativePath = $"{BaseDatasetsRelativePath}/wikiDetoxAnnotated40kRows.tsv";

        private static readonly string DataPath = GetAbsolutePath(DataRelativePath);

        private static readonly string BaseModelsRelativePath = @"../../../../MLModels";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/SentimentModel.zip";

        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {
            #region try
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);
            
            var loadedModels = new List<ITransformer>();
            for (var i = 0; i < 13; i++)
            {
                var model = mlContext.Model.Load(@"C:\src\MlDotNetSamples\samples\csharp\getting-started\BinaryClassification_SentimentAnalysis\SentimentAnalysis\MLModels\SentimentModel.zip", out var inputSchema);
                loadedModels.Add(model);
            }
            Console.ReadLine();
            #endregion
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath , relativePath);

            return fullPath;
        }
    }
}