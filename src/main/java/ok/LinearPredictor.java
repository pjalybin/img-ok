package ok;

import org.jblas.DoubleMatrix;

/**
* @author Petr Zhalybin
* @since 31.08.2014 13:37
*/
class LinearPredictor implements Predictor {
    private final DoubleMatrix theta;
    private final DoubleMatrix mean;
    private final DoubleMatrix spread;
    private final Parameters parameters;

    LinearPredictor(DoubleMatrix theta, DoubleMatrix mean, DoubleMatrix spread, Parameters parameters) {
        this.theta = theta;
        this.mean = mean;
        this.spread = spread;
        this.parameters = parameters;
    }

    @Override
    public double predict(Post post) {
        double[] f = new double[parameters.featuresLength];
        App.fillFeatures(f, post, parameters);
        DoubleMatrix X = new DoubleMatrix(f).sub(mean).div(spread);
        double logRes = X.mul(theta).sum();
        double res = Math.exp(logRes) - 1;
        if (res < 0) res = 0;
        return res;

    }

    public DoubleMatrix getSpread() {
        return spread;
    }

    public DoubleMatrix getMean() {
        return mean;
    }

    public DoubleMatrix getTheta() {
        return theta;
    }
}
