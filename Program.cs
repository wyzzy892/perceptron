using System;

namespace BackPropagationXor
{
    class Program
    {
        static void Main(string[] args)
        {
            train();
        }

        class sigmoid
        {
            public static double output(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }

            public static double derivative(double y)
            {
                return y * (1 - y);
            }
        }

        class Neuron
        {
            public double[] inputs = new double[2];
            public double[] weights = new double[2];
            public double error;

            private double biasWeight;

            private Random r = new Random();

            public double output
            {
                get { return sigmoid.output(weights[0] * inputs[0] + weights[1] * inputs[1] + biasWeight); }
            }

            public void randomizeWeights()
            {
                weights[0] = r.NextDouble();
                weights[1] = r.NextDouble();
                biasWeight = r.NextDouble();
            }

            public void adjustWeights()
            {
                weights[0] += error * inputs[0];
                weights[1] += error * inputs[1];
                biasWeight += error;
            }
        }

        private static void train()
        {
            // the input values
            double[,] inputs =
            {
                { 0, 0},
                { 0, 1},
                { 1, 0},
                { 1, 1}
            };

            // desired results
            double[] results = { 0, 1, 1, 0 };

            // creating the neurons
            Neuron hiddenNeuron1 = new Neuron();
            Neuron hiddenNeuron2 = new Neuron();
            Neuron outputNeuron = new Neuron();

            // random weights
            hiddenNeuron1.randomizeWeights();
            hiddenNeuron2.randomizeWeights();
            outputNeuron.randomizeWeights();

            int epoch = 0;

        Retry:
            epoch++;
            for (int i = 0; i < 4; i++)
            {
                // 1) forward propagation
                hiddenNeuron1.inputs = new double[] { inputs[i, 0], inputs[i, 1] };
                hiddenNeuron2.inputs = new double[] { inputs[i, 0], inputs[i, 1] };

                outputNeuron.inputs = new double[] { hiddenNeuron1.output, hiddenNeuron2.output };

                Console.WriteLine("{0} xor {1} = {2}", inputs[i, 0], inputs[i, 1], outputNeuron.output);

                // 2) back propagation

                // adjusts the weight of the output neuron
                outputNeuron.error = sigmoid.derivative(outputNeuron.output) * (results[i] - outputNeuron.output);
                outputNeuron.adjustWeights();

                // then adjusts neurons
                hiddenNeuron1.error = sigmoid.derivative(hiddenNeuron1.output) * outputNeuron.error * outputNeuron.weights[0];
                hiddenNeuron2.error = sigmoid.derivative(hiddenNeuron2.output) * outputNeuron.error * outputNeuron.weights[1];

                hiddenNeuron1.adjustWeights();
                hiddenNeuron2.adjustWeights();
            }

            if (epoch < 10000)
                goto Retry;

            Console.ReadLine();
        }
    }
}