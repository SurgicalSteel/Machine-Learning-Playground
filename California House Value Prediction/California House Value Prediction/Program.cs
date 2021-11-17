using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;

namespace California_House_Value_Prediction
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var dataset = mlContext
                            .Data
                            .LoadFromTextFile<HousingData>(
                                "./housing.csv", 
                                hasHeader: true, 
                                separatorChar: ','
                            );

            var split = mlContext.Data.TrainTestSplit(dataset, testFraction: 0.15);
            var rawTrainFeatures = split.TrainSet.Schema;

            var trainFeatures = (
                                    from column in rawTrainFeatures
                                    where column.Name != "Label" && column.Name != "OceanProximity"
                                    select column.Name
                                ).ToArray();

            var mlPipeline = mlContext.Transforms.Text.FeaturizeText("Text", "OceanProximity")
                                .Append(mlContext.Transforms.Concatenate("Features", trainFeatures))
                                .Append(mlContext.Transforms.Concatenate("Feature", "Features", "Text"))
                                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

            var model = mlPipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);

            var metrics = mlContext.Regression.Evaluate(predictions);

            Console.WriteLine("R^2 --> {0:0.00000}", metrics.RSquared);
        }
    }
}
