using CharDecoder.MachineLearning;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace CharDecoder
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        static WriteableBitmap writeableBitmap;
        static int columnCount = 28;
        static int rowCount = columnCount;
        static double finalDpi = (0.5) * columnCount;
        static double dpiDivisor = 96.0 / finalDpi;



        public MainWindow()
        {
            InitializeComponent();

            RenderOptions.SetBitmapScalingMode(image, BitmapScalingMode.NearestNeighbor);
            RenderOptions.SetEdgeMode(image, EdgeMode.Aliased);

            writeableBitmap = new WriteableBitmap(
                columnCount,
                rowCount,
                finalDpi,
                finalDpi,
                PixelFormats.Bgr32,
                null);

            image.Source = writeableBitmap;

            image.Stretch = Stretch.None;
            image.HorizontalAlignment = HorizontalAlignment.Left;
            image.VerticalAlignment = VerticalAlignment.Top;

            image.MouseMove += new MouseEventHandler(i_MouseMove);
            image.MouseLeftButtonDown +=
                new MouseButtonEventHandler(i_MouseLeftButtonDown);
            image.MouseRightButtonDown +=
                new MouseButtonEventHandler(i_MouseRightButtonDown);

        }

        void DrawPixel(MouseEventArgs e, byte[] colorData)
        {
            var x = (int)(e.GetPosition(image).X / dpiDivisor);
            var y = (int)(e.GetPosition(image).Y / dpiDivisor);

            if (x < 0 || columnCount <= x)
            {
                return;
            }

            if (y < 0 || rowCount <= y)
            {
                return;
            }

            Int32Rect rect = new Int32Rect(x, y, 1, 1);

            writeableBitmap.WritePixels(rect, colorData, 4, 0);
        }

        void i_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            DrawPixel(e, [0, 0, 0, 0]);
        }

        void i_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            DrawPixel(e, [255, 255, 255, 0]);
        }

        void i_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                DrawPixel(e, [255, 255, 255, 0]);
            }
            else if (e.RightButton == MouseButtonState.Pressed)
            {
                DrawPixel(e, [0, 0, 0, 0]);
            }
        }

        private void ButtonAsk_Click(object sender, RoutedEventArgs e)
        {
            if (!Trainer.IsDataTrained)
            {
                MessageBox.Show("Machine is not trained yet. Do the training first.");

                return;
            }

            int stride = writeableBitmap.PixelWidth * writeableBitmap.Format.BitsPerPixel / 8;
            int size = stride * writeableBitmap.PixelHeight;

            byte[] buffer = new byte[size];
            writeableBitmap.CopyPixels(buffer, stride, 0);

            var inputSize = size / 4;

            double[] buffer2 = new double[inputSize];

            for (int i = 0; i < inputSize; i++)
            {
                if (buffer[i * 4] == 0)
                {
                    buffer2[i] = 1.0;
                }
                else
                {
                    buffer2[i] = 0.0;
                }
            }

            var result = Trainer.Check(buffer2);

            MessageBox.Show($"Result: {result}");
        }

        private void ButtonReset_Click(object sender, RoutedEventArgs e)
        {
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    Int32Rect rect = new Int32Rect(j, i, 1, 1);
                    writeableBitmap.WritePixels(rect, new byte[] { 0, 0, 0, 0 }, 4, 0);
                }
            }
        }

        private void ButtonTrain_Click(object sender, RoutedEventArgs e)
        {
            Console.WriteLine("Getting data");
            MNIST.GetData();
            Console.WriteLine("Data read");

            Console.WriteLine("Training");
            Trainer.Train();
        }
    }
}