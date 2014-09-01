package ok;

import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.TrainingContinuation;
import org.jblas.DoubleMatrix;

/**
 * @author Petr Zhalybin
 * @since 31.08.2014 14:14
 */
public class NNPredictor implements Predictor {
    private final DoubleMatrix mean;
    private final DoubleMatrix spread;
    private final BasicNetwork nn;
    private final Parameters parameters;
    private final TrainingContinuation state;

    public NNPredictor(BasicNetwork network, DoubleMatrix mean, DoubleMatrix spread, Parameters parameters, TrainingContinuation state) {
        this.nn = network;
        this.mean = mean;
        this.spread = spread;
        this.parameters = parameters;
        this.state = state;
    }

    @Override
    public double predict(Post post) {
        double[] f = new double[parameters.featuresLength];
        App.fillFeatures(f, post, parameters);
        DoubleMatrix x = new DoubleMatrix(f).sub(mean).div(spread);
        double[] y = new double[1];
        nn.compute(x.data,y);
        return y[0];
    }

    public BasicNetwork getNn() {
        return nn;
    }

    public DoubleMatrix getMean() {
        return mean;
    }

    public DoubleMatrix getSpread() {
        return spread;
    }

    public TrainingContinuation getState() {
        return state;
    }
}
