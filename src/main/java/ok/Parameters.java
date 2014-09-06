package ok;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

/**
* @author Petr Zhalybin
* @since 31.08.2014 13:37
*/
class Parameters implements Serializable, Cloneable {


    final public String imagesWeightFolder = System.getProperty("ok.imagedir");
    final public String sentence2vecFile = System.getProperty("ok.sentence2vec");
    public Map<Long, App.ImageWeight[]> imagesWeights;
    final public boolean sortWords = Boolean.getBoolean("ok.sortWords");

    final int wordLimit = Integer.getInteger("ok.wordlimit", 1000);
    final int grouplimit = Integer.getInteger("ok.grouplimit", 100);
    final int wordFreqLimit = Integer.getInteger("ok.wordfreqlimit", 10);
    final int trigramFreqLimit = Integer.getInteger("ok.trigramfreqlimit", 100);
    final int trigramLimit = Integer.getInteger("ok.trigramlimit", 1000);

    public int bowDim;
    public int trigramDim;
    public int groupDim;
    public int sentence2vecDim;
    public final int timeDim = 24 + 7 + 12 + 10;
    public int featuresLength = -1;
    public long jan_01_2014;

    TreeMap<Integer, DateStat> totalDateStat;
    public List<String> words;
    public List<String> trigrams;
    public int[] groups;

    double learningRate = getDoubleProperty("ok.learningRate", 0.1);
    double regularization = getDoubleProperty("ok.regularization", 0);
    double learningRateFactor = getDoubleProperty("ok.learningRateFactor", 1);
    int minibatchSize = Integer.getInteger("ok.minibatchSize", 10000);

    final boolean noMapping = Boolean.getBoolean("ok.noMapping");
    boolean tfidfBow = Boolean.getBoolean("ok.tfidf");
    int maxbatchSize = Integer.getInteger("ok.maxbatchSize", 100000);
    double minibatchFactor = getDoubleProperty("ok.minibatchFactor", 1);
    public int imageDim = imagesWeightFolder == null ? 0 : 1000;
    boolean[] skipColumns;
    public App.ImageClass[] imgCls;
    public final double logScale = getDoubleProperty("ok.logScale", 20);
    public final boolean bigrams=Boolean.getBoolean("ok.bigrams");
    public final Random rnd = new Random(Long.getLong("ok.seed", 42));

    public int[] nnLayers=null;
    public final double nnMomentum = getDoubleProperty("ok.nn.momentum", 0);
    public final int nnIter = Integer.getInteger("ok.nn.iter", 1);
    public final int nnBatch = Integer.getInteger("ok.nn.batch", 10000);
    public final int nnHyp=Integer.getInteger("ok.nn.hyp",-1);

    public final double devSetFrac = getDoubleProperty("ok.devset", 0.2);


    private double getDoubleProperty(String propertyName, double defaultValue) {
        String val = System.getProperty(propertyName);
        return val == null ? defaultValue : Double.valueOf(val);
    }

    @Override
    public Parameters clone() {
        try {
            return (Parameters)super.clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
