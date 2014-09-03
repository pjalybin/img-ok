package ok;

import de.jungblut.classification.nn.MultilayerPerceptron;
import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;

import java.io.*;

/**
 * @author Petr Zhalybin
 * @since 31.08.2014 14:14
 */
public class NNPredictor implements Predictor {
    private final DoubleVector mean;
    private final DoubleVector spread;
    private final MultilayerPerceptron nn;
    private final Parameters parameters;

    public NNPredictor(MultilayerPerceptron network, DoubleVector mean, DoubleVector spread, Parameters parameters) {
        this.nn = network;
        this.mean = mean;
        this.spread = spread;
        this.parameters = parameters;
    }

    public NNPredictor clone(){
            try(ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                ObjectOutputStream stream = new ObjectOutputStream(bytes)) {
                MultilayerPerceptron.serialize(nn, stream);
                try(DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
                    MultilayerPerceptron copy = MultilayerPerceptron.deserialize(in);
                    NNPredictor clone = new NNPredictor(copy,mean,spread,parameters);
                    return clone;
                }
            } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public double predict(Post post) {
        double[] f = new double[parameters.featuresLength];
        App.fillFeatures(f, post, parameters);
        DoubleVector x = new DenseDoubleVector(f).subtract(mean).divide(spread);
        double y = nn.predict(x).get(0);
        if(y<0)y=0;
        return y;
    }

    public MultilayerPerceptron getNn() {
        return nn;
    }

    public DoubleVector getMean() {
        return mean;
    }

    public DoubleVector getSpread() {
        return spread;
    }

}
