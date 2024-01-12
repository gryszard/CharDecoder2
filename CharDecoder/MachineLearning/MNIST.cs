using System.IO;
using System.Windows.Media;
using NumSharp;

namespace CharDecoder.MachineLearning
{
    public static class MNIST
    {
        public static List<int> Labels { get; set; } = new();
        public static List<List<double>> Pixels { get; set; } = new();
        public static List<NDArray> NDLabels { get; set; } = new();


        public static void GetData()
        {
            using (var reader = new StreamReader(@"./Resources/mnist_train.csv"))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');

                    var label = int.Parse(values[0]);

                    Pixels.Add(values.Skip(1).Select(x => int.Parse(x) / 255.0).ToList());

                    Labels.Add(label);
                    NDLabels.Add(np.eye(10)[label]);
                }
            }
        }
    }

    public static class Trainer
    {
        private static NDArray W_I_H { get; set; }
        private static NDArray W_H_O { get; set; }
        private static NDArray B_I_H { get; set; }
        private static NDArray B_H_O { get; set; }

        public static int Check(double[] pixels)
        {
            var image2 = np.reshape(pixels, (28*28, 1));

            // Forward propagation input -> hidden
            var h_pre = B_I_H + np.matmul(W_I_H, image2);
            var h = 1 / (1 + np.exp(-1 * h_pre));

            // Forward propagation hidden -> output
            var o_pre = B_H_O + np.matmul(W_H_O, h);
            var o = 1 / (1 + np.exp(-1 * o_pre));

            var result = np.argmax(o);

            return result;
        }

        public static void Train()
        {
            W_I_H = np.random.uniform(-0.5, 0.5, (20, 784));
            W_H_O = np.random.uniform(-0.5, 0.5, (10, 20));
            B_I_H = np.zeros((20, 1));
            B_H_O = np.zeros((10, 1));

            double learnRate = 0.01;
            int epochs = 3;
            
            for (int e = 0; e < epochs; e++)
            {
                Console.WriteLine($"Epoc: {e + 1}");
                int nrCorrect = 0;

                for (int img = 0; img < MNIST.Pixels.Count; img++)
                {
                    var image = MNIST.Pixels[img].ToArray();
                    var image2 = np.reshape(image, (784, 1));

                    var labels = MNIST.NDLabels[img];
                    labels = np.reshape(labels, (10, 1));

                    // Forward propagation input -> hidden
                    var h_pre = B_I_H + np.matmul(W_I_H, image2);
                    var h = 1 / (1 + np.exp(-1 * h_pre));

                    // Forward propagation hidden -> output
                    var o_pre = B_H_O + np.matmul(W_H_O, h);
                    var o = 1 / (1 + np.exp(-1 * o_pre));

                    // Cost / Error calculation.
                    var t1 = np.power((o - labels), 2);
                    var suma = t1.ToArray<double>().Sum();
                    var error = 1.0 / (o.size) * suma;

                    // Counting correct matches.
                    if (np.argmax(o) == np.argmax(labels))
                    {
                        nrCorrect++;
                    }

                    // Backpropagation output -> hidden (cost function derivative)
                    var delta_o = o - labels;
                    W_H_O += -learnRate * np.matmul(delta_o, np.transpose(h));
                    B_H_O += -learnRate * delta_o;

                    // Backpropagation hidden -> input (activation function derivative)
                    var delta_h = np.matmul(np.transpose(W_H_O), delta_o) * (h * (1 - h));
                    W_I_H += -learnRate * np.matmul(delta_h, np.transpose(image2));
                    B_I_H += -learnRate * delta_h;

                    if (img % 10000 == 0)
                    {
                        Console.WriteLine($"Epoc: {e + 1}: img: {img} / {MNIST.Pixels.Count}");
                    }
                }

                var acc = 1.0 * nrCorrect / MNIST.Pixels.Count;
                Console.WriteLine($"Acc [{e}] = {(acc * 100)} %");
            }
        }
    }
}