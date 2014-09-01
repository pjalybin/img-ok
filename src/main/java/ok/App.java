package ok;

import au.com.bytecode.opencsv.CSVReader;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.randomize.ConsistentRandomizer;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.TrainingContinuation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.Eigen;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.RangeUtils;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.russianStemmer;

import java.io.*;
import java.util.*;
import java.util.concurrent.RecursiveAction;

/**
 * @author Petr Zhalybin
 * @since 06.07.2014
 */
public class App {

    public static void main(String[] args) throws IOException {

        final Parameters parameters = new Parameters();

        final Map<Integer, Post> posts = readPosts("train_content.csv");

        final Map<Integer, Post> testPosts = readPosts("test_content.csv");

        if (parameters.sentence2vecFile != null) {
            parameters.sentence2vecDim = readSentence2vec(new File(parameters.sentence2vecFile), posts, testPosts);
        }

        // printSent2vecClosestPosts(posts, testPosts, 100);

        readLikes(posts);

        printBestPosts(posts, 100);

        TreeMap<Integer, DateStat> totalDateStat = new TreeMap<Integer, DateStat>();

        Map<Integer, GroupStat> groupStatMap = countGroups(posts, totalDateStat);
        ArrayList<GroupStat> groupStats = new ArrayList<GroupStat>(groupStatMap.values());
        Collections.sort(groupStats, new Comparator<GroupStat>() {
            @Override
            public int compare(GroupStat o1, GroupStat o2) {
                return Double.compare(
                        o1.likesStat.likesum / o1.likesStat.count,
                        o2.likesStat.likesum / o2.likesStat.count
                );
            }
        });

        for (GroupStat w : groupStats) {
            System.out.println(w.groupid + " " + w.likesStat.likesum / w.likesStat.count);

        }

        readLikes2(posts, totalDateStat);

        computePostGroupTimeStat(posts, testPosts);


        parameters.totalDateStat = totalDateStat;

        if (parameters.imagesWeightFolder != null) {
            File dirImages = new File(parameters.imagesWeightFolder);
            parameters.imagesWeights = readImages(dirImages, new File(dirImages, "imagenet_classe.txt"), parameters);
        }


        parseBow(posts, testPosts, parameters);


        for (Post post : testPosts.values()) {
            post.groupStat = groupStatMap.get(post.groupid);
        }

        parameters.groupDim = 0;
        for (GroupStat groupStat : groupStats) {
            groupStat.bowid = parameters.groupDim++;
            if (parameters.groupDim >= parameters.grouplimit) break;
        }

        parameters.groups = new int[parameters.groupDim];
        int gn = 0;
        for (GroupStat groupStat : groupStats) {
            if (groupStat.bowid >= 0) {
                if (gn >= parameters.grouplimit) break;
                parameters.groups[gn++] = groupStat.groupid;
            }
        }

        //Writer writer = new BufferedWriter(new FileWriter("features.csv"));

        parameters.featuresLength = 45 * (parameters.noMapping ? 1 : 4) + 12
                + parameters.bowDim
                + parameters.trigramDim
                + parameters.groupDim
                + parameters.timeDim
                + parameters.imageDim
                + parameters.sentence2vecDim
                + 2;

        System.out.println("Features length " + parameters.featuresLength);

        setSkipcolumns(parameters);

        Calendar calendar = new GregorianCalendar(2014, 0, 1, 0, 0, 0);
        calendar.setTimeZone(TimeZone.getTimeZone("GMT"));
        parameters.jan_01_2014 = calendar.getTimeInMillis();


        List<Post> shuffled = new ArrayList<>(posts.values());

        Collections.shuffle(shuffled, parameters.rnd);
        int devIdx = (int) (shuffled.size() * parameters.devSetFrac);
        List<Post> dev = new ArrayList<>(shuffled.subList(0, devIdx));
        List<Post> train = new ArrayList<>(shuffled.subList(devIdx, shuffled.size()));

        String saveTrainFeaturesFile = System.getProperty("ok.saveTrainFeaturesFile");
        String saveFeaturesFileFormat = System.getProperty("ok.saveFeaturesFileFormat", "CSV").toUpperCase();
        PostFeaturesSaver featuresSaver = PostFeaturesSaver.valueOf(saveFeaturesFileFormat);
        final Map<Integer, List<Post>> groups = getGroups(train, Integer.getInteger("ok.saveGroupsLimit", 500));
        final Map<Integer, List<Post>> groups2 = getGroups(posts.values(), Integer.getInteger("ok.saveGroupsLimit", 500));
        if (saveTrainFeaturesFile != null) {
            featuresSaver.saveFeatures(posts.values(), parameters, new File(saveTrainFeaturesFile));
            featuresSaver.saveFeatures(dev, parameters, new File(saveTrainFeaturesFile + ".dev." + featuresSaver.name().toLowerCase()));
            featuresSaver.saveFeatures(train, parameters, new File(saveTrainFeaturesFile + ".train." + featuresSaver.name().toLowerCase()));

            File dir = new File(saveTrainFeaturesFile + ".groups");
            saveGroups(groups2, parameters, dir, featuresSaver);
            saveFeaturesNames(parameters, new File(saveTrainFeaturesFile + ".features.csv"));
        }
        String saveTestFeaturesFile = System.getProperty("ok.saveTestFeaturesFile");
        if (saveTestFeaturesFile != null) {
            featuresSaver.saveFeatures(testPosts.values(), parameters, new File(saveTestFeaturesFile));
        }

        String nn = System.getProperty("ok.nn");
        if(nn!=null){
            String[] nns=nn.split(",");
            int[] nnl=new int[nns.length];
            for (int i = 0; i < nnl.length; i++) {
                 nnl[i]=Integer.valueOf(nns[i].trim());
            }
            parameters.nnLayers=nnl;
        }

        Trainer trainer = parameters.nnLayers!=null ? new Trainer() {
            @Override
            public Predictor train(Collection<Post> posts, Parameters parameters, String trainId, Map<Integer, Post> testPosts, Predictor initialPred, int epochNum, Map<Integer, Predictor> groupPred, int groupPredKey, Collection<Post> dev) {
                return trainNN(posts, parameters, trainId, testPosts, initialPred, epochNum, groupPred, groupPredKey, dev);
            }
        } : new Trainer() {
            @Override
            public Predictor train(Collection<Post> posts, Parameters parameters, String trainId, Map<Integer, Post> testPosts, Predictor initialPred, int epochNum, Map<Integer, Predictor> groupPred, int groupPredKey, Collection<Post> dev) {
                return trainLinear(posts, parameters, trainId, testPosts, initialPred, epochNum, groupPred, groupPredKey);
            }
        };

        trainGropusParallel(posts.values(), train, dev, testPosts, parameters, groups, trainer);

        System.out.println("Done.");

    }

    private static void printSent2vecClosestPosts(Map<Integer, Post> posts, Map<Integer, Post> testPosts, int limit) {
        List<Post> postsShuffled = new ArrayList<>(posts.values());
        Collections.shuffle(postsShuffled);
        postsShuffled = postsShuffled.subList(0, limit);
        for (Post post : postsShuffled) {
            System.out.println(post.txt);
            List<Post> closestSent2vec = findClosestSent2vec(post, 10, posts, testPosts);
            for (Post p : closestSent2vec.subList(1, 10)) {
                System.out.println("              " + p.txt);
            }
        }
    }

    private static int readSentence2vec(File file, Map<Integer, Post>... posts) throws IOException {
        System.out.println("Reading Sentence2vec");
        try (CSVReader reader = new CSVReader(new FileReader(file), ' ', '\0')) {
            String[] head = reader.readNext();
            int rows = Integer.valueOf(head[0]);
            int sum = 0;
            for (Map<Integer, Post> postMap : posts) {
                sum += postMap.size();
            }
//            if (rows != sum) throw new RuntimeException("Different size " + rows + " " + sum);
            int dim = Integer.valueOf(head[1]);
            int ii = 0;
            for (Map<Integer, Post> postMap : posts) {
                for (Map.Entry<Integer, Post> e : postMap.entrySet()) {
                    if (++ii % 10000 == 0) System.out.println(ii);
                    Post post = e.getValue();
                    String[] row = reader.readNext();
                    if (row.length != dim + 1) throw new RuntimeException("Bad row " + row.length);
                    double[] sentence2vec = new double[dim];
                    for (int i = 1; i < row.length; i++) {
                        sentence2vec[i - 1] = Double.valueOf(row[i]);
                    }
                    post.sentence2vec = sentence2vec;
                }
            }
            return dim;
        }
    }

    private static void parseBow(Map<Integer, Post> posts, Map<Integer, Post> testPosts, Parameters parameters) {
        Tokeniser tokeniser = new TokenizerCharType();
        Map<String, WordStat> wordStatHashMap = new HashMap<>();
        Map<String, WordStat> trigramsStatHashMap = new HashMap<>();

        try {
            parseWords(tokeniser, wordStatHashMap, trigramsStatHashMap, parameters.bigrams, posts, testPosts);
        } catch (IOException e) {
            e.printStackTrace();
        }

        parameters.words = prepareBow(wordStatHashMap, posts.size(), parameters.wordFreqLimit, parameters.wordLimit);
        parameters.bowDim = parameters.words.size();
        System.out.println("BOW size " + parameters.bowDim);

        parameters.trigrams = prepareBow(trigramsStatHashMap, posts.size(), parameters.trigramFreqLimit, parameters.trigramLimit);
        parameters.trigramDim = parameters.trigrams.size();
        System.out.println("Bag of trigrams size " + parameters.trigramDim);
    }

    private static void saveFeaturesNames(Parameters parameters, File file) throws IOException {
        try (PrintWriter filewriter = new PrintWriter(file)) {
            String[] fnames = getFeatureNames(parameters);
            for (String f : fnames) {
                String fr = getFeatureReadableName(parameters, f);
                if (f.equals(fr)) {
                    filewriter.println(f);
                } else {
                    filewriter.println(f + "," + fr);
                }
            }
        }
    }


    private static void trainGropusParallel(final Collection<Post> all,
                                            final Collection<Post> train,
                                            final Collection<Post> dev,
                                            final Map<Integer, Post> testPosts,
                                            final Parameters parameters,
                                            final Map<Integer, List<Post>> groups,
                                            Trainer trainer) throws IOException {
        final Predictor predictorAll = trainer.train(train, parameters, "all", testPosts, null,
                Integer.getInteger("ok.epochnum2", 5), null, 0, dev);


        test(dev, predictorAll);
//        savePredictor(predictorAll);

        final Map<Integer, Predictor> groupPred = new TreeMap<>();
        ArrayList<RecursiveAction> actions = new ArrayList<>();


        for (final Map.Entry<Integer, List<Post>> e : groups.entrySet()) {
            RecursiveAction action = new RecursiveAction() {
                @Override
                protected void compute() {
                    int g = e.getKey();
                    List<Post> po = e.getValue();
                    Predictor predictor = trainer.train(po, parameters, "Group" + g, null, predictorAll,
                            Integer.getInteger("ok.epochnum", 100), groupPred, g, dev);
                    synchronized (groupPred) {
                        groupPred.put(g, predictor);
                    }

                }
            };
            action.fork();
            actions.add(action);
        }

        Timer timer = new Timer(true);
        long period = 1000 * 60 * 30; //30 min
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                Predictor ensembleGroupPredictor = null;
                synchronized (groupPred) {
                    if (groupPred.size() == groups.size()) {
                        ensembleGroupPredictor = new PredictorGroupEnsemble(groupPred);
                    }
                }
                if (ensembleGroupPredictor != null) {
                    try {
                        test(dev, ensembleGroupPredictor);
                        saveResults(ensembleGroupPredictor, all, testPosts.values());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }
        }, period, period);


        for (RecursiveAction action : actions) {

            action.join();
        }
        timer.cancel();


        Predictor ensembleGroupPredictor = new PredictorGroupEnsemble(groupPred);

        test(dev, ensembleGroupPredictor);

        saveResults(ensembleGroupPredictor, all, testPosts.values());
    }

    private static double[] test(Collection<Post> dev, Predictor predictor) {
        double[] pm = new double[dev.size()];
        double[] ym = new double[dev.size()];
        int i = 0;
        for (Post post : dev) {
            pm[i] = predictor.predict(post);
            ym[i] = post.likes;
            i++;
        }
        DoubleMatrix p = new DoubleMatrix(i, 1, pm);
        DoubleMatrix y = new DoubleMatrix(i, 1, ym);
        DoubleMatrix sub = p.sub(y);
        double v1 = variance(sub).sum();
        double v2 = variance(y).sum();
        double r2 = v2 < 1e-6 ? 0 : (1 - v1 / v2) * 1000;
        double cost = MatrixFunctions.pow(sub, 2).sum() / i;
        System.out.println("test R2 = " + r2+"\t Cost = " + cost);
        return new double[]{r2, cost};
    }

    private static void saveResults(Predictor predictor, Collection<Post> train, Collection<Post> test) throws IOException {
        String fileName = System.getProperty("ok.outcsv", "test_result.csv");
        saveResults(predictor, fileName.replace(".csv", "_train.csv"), train, true);
        saveResults(predictor, fileName, test, false);
        savePredictor(predictor);
        System.out.println("Saved result " + new Date());
    }

    private static List<String> prepareBow(Map<String, WordStat> wordStatHashMap, int postsSize, int wordFreqLimit, int wordLimit) {
        ArrayList<WordStat> wordStats = new ArrayList<>(wordStatHashMap.values());

        Comparator<? super WordStat> comparator = (o1, o2) -> -Double.compare(
                o1.likesStat.likesum / o1.likesStat.count,
                o2.likesStat.likesum / o2.likesStat.count
        );

        Collections.sort(wordStats, comparator);


        List<String> words = new ArrayList<>();

        for (WordStat wordStat : wordStats) {
            int c = wordStat.likesStat.count;
            wordStat.idf = Math.log((double) postsSize / c);
            if (c >= wordFreqLimit) {
                words.add(wordStat.word);
            }
            if (words.size() >= wordLimit) {
                break;
            }
        }
        System.out.println(words.toString());
        int bowDim = words.size();
        for (int i = 0; i < bowDim; i++) {
            String w = words.get(i);
            WordStat wordStat = wordStatHashMap.get(w);
            wordStat.bowId = i;
        }
        return words;
    }

    private static void saveGroups(Map<Integer, List<Post>> postsByGroupTime, Parameters parameters, File dir, PostFeaturesSaver saver) throws IOException {
        dir.mkdirs();

        for (Map.Entry<Integer, List<Post>> e : postsByGroupTime.entrySet()) {
            saver.saveFeatures(e.getValue(), parameters, new File(dir, "g" + e.getKey() + "." + saver.name().toLowerCase()));
        }
    }

    private static void setSkipcolumns(Parameters parameters) {
        String skipColumns = System.getProperty("ok.onlyColumns");
        boolean skip = true;
        if (skipColumns == null) {
            skipColumns = System.getProperty("ok.skipColumns");
            skip = false;
        }
        if (skipColumns != null) {
            String[] skipColStr = skipColumns.split(",");
            HashSet<String> set = new HashSet<>();
            for (String s : skipColStr) {
                set.add(s.trim());
            }
            parameters.skipColumns = getSkipcolumns(set, parameters, skip);
        }
    }

    private static boolean[] getSkipcolumns(Set<String> columnsNames, Parameters parameters, boolean skip) {
        String[] fn = getFeatureNames(parameters);
        boolean[] res = new boolean[fn.length];
        for (int i = 0; i < fn.length; i++) {
            res[i] = columnsNames.contains(fn[i]) ^ skip;
            if (!res[i]) {
                System.out.println(fn[i] + "\t" + getFeatureReadableName(parameters, fn[i]));
            }
        }
        return res;
    }

    private static String getFeatureReadableName(Parameters parameters, String fname) {
        String word_ = "word_";
        String image_ = "image_";
        String trigram_ = "trigram_";
        if (fname.startsWith(trigram_)) {
            fname = trigram_ + parameters.trigrams.get(Integer.valueOf(fname.substring(trigram_.length())));
        } else if (fname.startsWith(word_)) {
            fname = word_ + parameters.words.get(Integer.valueOf(fname.substring(word_.length())));
        } else if (fname.startsWith(image_)) {
            int img_id = Integer.valueOf(fname.substring(image_.length()));
            ImageClass imageClass = parameters.imgCls[img_id];
            fname = image_ + imageClass.name.replace(',', '-').replace(' ', '_');
        }
        return fname;
    }

    private static Map<Integer, List<Post>> getGroups(Collection<Post> posts, int groupslimit) {
        final Map<Integer, List<Post>> postsByGroupTime = new TreeMap<>();
        for (Post post : posts) {
            int groupid = post.groupid;
            if (post.groupStat.count < groupslimit) {
                groupid = 0;
            }
            List<Post> postsInGroup = postsByGroupTime.get(groupid);
            if (postsInGroup == null) {
                postsInGroup = new ArrayList<Post>();
                postsByGroupTime.put(groupid, postsInGroup);
            }
            postsInGroup.add(post);

        }
        return postsByGroupTime;
    }


    private static void saveResults(Predictor predictor, String fileName, Collection<Post> posts, boolean likeCol) throws IOException {
        try (PrintWriter filewriter = new PrintWriter(new FileWriter(fileName))) {
            for (Post post : posts) {
                double predictLikes = predictor.predict(post);
                StringBuilder sb = new StringBuilder();
                sb.append(post.id).append(',').append(predictLikes);
                if (likeCol) sb.append(',').append(post.likes);
                filewriter.println(sb.toString());
            }
        }
    }

    private static void savePredictor(Predictor predictor) throws IOException {
        String file = System.getProperty("ok.outfile", "predictor.bin");
        System.out.println("Save result to " + file);
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(file))) {
            objectOutputStream.writeObject(predictor);
        }
    }

    public static String[] getFeatureNames(Parameters parameters) {
        String[] fn = new String[parameters.featuresLength];
        String[] day = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
        fn[0] = "zero";
        fn[1] = "one";
        int j = 2;
        for (int i = 0; i < parameters.bowDim; i++) {
            fn[j + i] = "word_" + i;//parameters.words.get(i);
        }
        j += parameters.bowDim;
        for (int i = 0; i < parameters.trigramDim; i++) {
            fn[j + i] = "trigram_" + i;//parameters.words.get(i);
        }
        j += parameters.trigramDim;
        for (int i = 0; i < parameters.groupDim; i++) {
            fn[j + i] = "group_" + parameters.groups[i];
        }
        j += parameters.groupDim;
        for (int i = 0; i < 24; i++) {
            fn[j + i] = "hour_" + i;
        }
        j += 24;
        for (int i = 0; i < 7; i++) {
            fn[j + i] = "day_" + day[i];
        }
        j += 7;
        for (int i = 0; i < 12; i++) {
            fn[j + i] = "month_" + (i + 1);
        }
        j += 12;
        for (int i = 0; i < 10; i++) {
            fn[j + i] = "year_" + (i + 2005);
        }
        j += 10;
        for (int i = 0; i < parameters.imageDim; i++) {
            fn[j + i] = "image_" + i;
        }
        j += parameters.imageDim;
        for (int i = 0; i < parameters.sentence2vecDim; i++) {
            fn[j + i] = "sent2vec_" + i;
        }
        j += parameters.sentence2vecDim;
        fn[j++] = "bool_voskl";
        fn[j++] = "bool_3voskl";
        fn[j++] = "bool_smile2";
        fn[j++] = "bool_smile3";
        fn[j++] = "bool_smile";
        fn[j++] = "bool_3smile"
        ;
        fn[j++] = "bool_Image";
        fn[j++] = "bool_Pool";
        fn[j++] = "bool_http";
        fn[j++] = "bool_youtube";
        fn[j++] = "bool_klass";
        fn[j++] = "bool_laik";

        int is = j;
        fn[j++] = "num_caps";
        fn[j++] = "num_caps_word";
        fn[j++] = "num_vosk";
        fn[j++] = "num_vosk_char";
        fn[j++] = "num_wordsn";

        fn[j++] = "num_length";
        fn[j++] = "num_age2014";
        fn[j++] = "num_images";
        fn[j++] = "num_pool";
        fn[j++] = "num_unknownwords";
        fn[j++] = "num_maxtf";

        fn[j++] = "num_grouplikesum";
        fn[j++] = "num_grouploglikesum";
        fn[j++] = "num_grouplikesumavg";
        fn[j++] = "num_grouploglikesumavg";

        fn[j++] = "num_frequencyForTenPostsInGroup";
        fn[j++] = "num_postsInGroupTomorrowDelta";
        fn[j++] = "num_postsInGroupHourDelta";

        fn[j++] = "num_yearage";
        fn[j++] = "num_month";
        fn[j++] = "num_dayofmonth";
        fn[j++] = "num_dayofweek";
        fn[j++] = "num_hour";

        fn[j++] = "num_month_sin";
        fn[j++] = "num_dayofmonth_sin";
        fn[j++] = "num_dayofweek_sin";
        fn[j++] = "num_hour_sin";

        fn[j++] = "num_month_cos";
        fn[j++] = "num_dayofmonth_cos";
        fn[j++] = "num_dayofweek_cos";
        fn[j++] = "num_hour_cos";

        fn[j++] = "num_dayposts_group";
        fn[j++] = "num_daylikes_group";
        fn[j++] = "num_daylikesperpost_group";
        fn[j++] = "num_monthpposts_group";
        fn[j++] = "num_monthlikes_group";
        fn[j++] = "num_monthlikesperpost_group";

        fn[j++] = "num_dayposts_total";
        fn[j++] = "num_daylikes_total";
        fn[j++] = "num_daylikesperpost_total";
        fn[j++] = "num_monthpposts_total";
        fn[j++] = "num_monthlikes_total";
        fn[j++] = "num_monthlikesperpost_total";

        fn[j++] = "num_trigramsNum";
        fn[j++] = "num_maxtrigramTf";


        if (!parameters.noMapping) {
            for (int i = 0; i < j - is; i++) {    //43
                String s = fn[is + i];
                fn[i + j] = "log_" + s;
                fn[i + j + (j - is)] = "sqr_" + s;
                fn[i + j + 2 * (j - is)] = "cube_" + s;
            }
        }
        if (fn[fn.length - 1] == null) throw new NullPointerException();
        return fn;
    }


    private static Predictor trainNN(Collection<Post> posts, Parameters parameters, String trainId,
                                     Map<Integer, Post> testPosts, Predictor initialPred, int epochNum,
                                     final Map<Integer, Predictor> groupPred, int groupPredKey,
                                     Collection<Post> dev) {

        int pn;


        double[] f;

        double learningRate = parameters.learningRate;
        int minibatchSize = parameters.minibatchSize;
        if (minibatchSize > posts.size()) {
            minibatchSize = posts.size();
        }

        int epoch = 0;

        int maxbatchSize = parameters.maxbatchSize;
        if (maxbatchSize > posts.size()) maxbatchSize = posts.size();
        int maxrows = Integer.MAX_VALUE / parameters.featuresLength;
        if (maxbatchSize > maxrows) maxbatchSize = maxrows;

        BasicNetwork network;
        DoubleMatrix maxF = null;
        DoubleMatrix minF = null;
        DoubleMatrix sumF = null;
        DoubleMatrix mean = null;
        DoubleMatrix spread = null;
        DoubleMatrix X;
        DoubleMatrix Y;

        TrainingContinuation state = null;

        if (initialPred instanceof NNPredictor) {
            NNPredictor nnPredictor = (NNPredictor) initialPred;
            network = (BasicNetwork) nnPredictor.getNn().clone();
            mean = nnPredictor.getMean();
            spread = nnPredictor.getSpread();
            state = nnPredictor.getState();
            epoch = 1;
        } else {
            network = new BasicNetwork();
            network.addLayer(new BasicLayer(null, true, parameters.featuresLength));
            for (int nnLayer : parameters.nnLayers) {
                network.addLayer(new BasicLayer(new ActivationTANH(), true, nnLayer));
            }
            network.addLayer(new BasicLayer(new ActivationLinear(), false, 1));

            network.getStructure().finalizeStructure();
            network.reset();
            new ConsistentRandomizer(-1, 1, 500).randomize(network);

        }
        X = new DoubleMatrix(minibatchSize, parameters.featuresLength);
        Y = new DoubleMatrix(minibatchSize, 1);

        ArrayList<Post> shuffledPosts = new ArrayList<>(posts);

        for (; epoch < epochNum; epoch++) {

            Collections.shuffle(shuffledPosts, parameters.rnd);
            pn = 0;

            List<Post> nextPosts=new ArrayList<>(minibatchSize);
            for (Post post : shuffledPosts) {

                pn++;
                f = new double[parameters.featuresLength];

                fillFeatures(f, post, parameters);
                if (epoch == 0) {
                    if (maxF == null) {
                        maxF = new DoubleMatrix(f);
                        minF = new DoubleMatrix(f);
                        sumF = new DoubleMatrix(f);
                    } else {
                        mean = new DoubleMatrix(f);
                        maxF = maxF.max(mean);
                        minF = minF.min(mean);
                        sumF = sumF.add(mean);
                    }
                } else {
                    nextPosts.add(post);

                    double y = post.likes;

                    int row = (pn-1) % minibatchSize;
                    X.putRow(row, new DoubleMatrix(f).sub(mean).div(spread));
                    Y.put(row, 0, y);

                    if (row + 1 == minibatchSize) {

//                        if(initialPred==null){
//                            System.out.println("nn "+trainId+" epoch="+epoch+" p="+pn);
//                            test(nextPosts, new NNPredictor(network, mean, spread, parameters));
//                        }

                        double[][] xa = X.toArray2();
                        double[][] ya = Y.toArray2();
                        BasicMLDataSet trainingSet = new BasicMLDataSet(xa, ya);
                        Backpropagation train = new Backpropagation(network, trainingSet, learningRate, parameters.nnMomentum);


                        train.fixFlatSpot(true);
                        if(state!=null){
                            train.resume(state);
                        }

                        train.iteration(parameters.nnIter);


                        double error = train.getError();
                        System.out.println("\t\t\t"+trainId + " epoch=" + epoch + " error=" + error);
                        if(initialPred==null && dev!=null){
                            test(dev, new NNPredictor(network, mean, spread, parameters, state));
                        }


                        state = train.pause();

                        nextPosts.clear();

                    }

                }
            }

            if (epoch == 0) {
                int size = shuffledPosts.size();
                mean = sumF.div(size);
                spread = maxF.sub(minF);
                for (int j = 0; j < parameters.featuresLength; j++) {
                    double s = spread.get(j);
                    if (s < 1e-6) {
                        spread.put(j, 1);
                        mean.put(j, 0);
                    }
                }


            } else {
                if (Math.abs(learningRate * parameters.learningRateFactor) < 1) {
                    learningRate = learningRate * parameters.learningRateFactor;
                }
                if (parameters.minibatchFactor > 1.0) {
                    minibatchSize = (int) Math.ceil(minibatchSize * parameters.minibatchFactor);
                }
                if (minibatchSize >= maxbatchSize) minibatchSize = maxbatchSize;
                if (testPosts != null || groupPred != null) {
                    try {
                        NNPredictor predictor = new NNPredictor(network, mean, spread, parameters, state);
                        if (groupPred != null) {
                            synchronized (groupPred) {
                                groupPred.put(groupPredKey, predictor);
                            }
                        }
                        if (testPosts != null) {
                            saveResults(predictor, posts, testPosts.values());
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }


        }


        return new NNPredictor(network, mean, spread, parameters, state);
    }


    private static Predictor trainLinear(Collection<Post> posts, Parameters parameters, String trainId,
                                         Map<Integer, Post> testPosts, Predictor initialPred, int epochNum,
                                         final Map<Integer, Predictor> groupPred, int groupPredKey) {


        int pn = 0;


        double[] f = new double[parameters.featuresLength];

        double learningRate = parameters.learningRate;
        int minibatchSize = parameters.minibatchSize;
        if (minibatchSize > posts.size()) {
            minibatchSize = posts.size();
        }


        DoubleMatrix X;
        DoubleMatrix Y;
        DoubleMatrix theta = new DoubleMatrix(parameters.featuresLength, 1);

        DoubleMatrix maxF = null;
        DoubleMatrix minF = null;
        DoubleMatrix sumF = null;
        DoubleMatrix mean = null;
        DoubleMatrix spread = null;
        int epoch = 0;
        if (initialPred instanceof LinearPredictor) {
            theta = ((LinearPredictor) initialPred).getTheta().dup();
            spread = ((LinearPredictor) initialPred).getSpread().dup();
            mean = ((LinearPredictor) initialPred).getMean().dup();
            epoch = 1;
        }

        int maxbatchSize = parameters.maxbatchSize;
        if (maxbatchSize > posts.size()) maxbatchSize = posts.size();
        int maxrows = Integer.MAX_VALUE / parameters.featuresLength;
        if (maxbatchSize > maxrows) maxbatchSize = maxrows;

        ArrayList<Post> shuffledPosts = new ArrayList<>(posts);
        for (; epoch < epochNum; epoch++) {

            X = new DoubleMatrix(minibatchSize, parameters.featuresLength);
            Y = new DoubleMatrix(minibatchSize, 1);

            Collections.shuffle(shuffledPosts, parameters.rnd);
            pn = 0;


            for (Post post : shuffledPosts) {

                pn++;

                fillFeatures(f, post, parameters);

                if (parameters.skipColumns != null) {
                    for (int i = 0; i < parameters.skipColumns.length; i++) {
                        if (parameters.skipColumns[i]) f[i] = 0;
                    }
                }

                if (epoch == 0) {
                    if (maxF == null) {
                        maxF = new DoubleMatrix(f);
                        minF = new DoubleMatrix(f);
                        sumF = new DoubleMatrix(f);
                    } else {
                        mean = new DoubleMatrix(f);
                        maxF = maxF.max(mean);
                        minF = minF.min(mean);
                        sumF = sumF.add(mean);
                    }
                } else {
                    double y = post.loglikes;

                    int row = (pn-1) % minibatchSize;
                    X.putRow(row, new DoubleMatrix(f).sub(mean).div(spread));
                    Y.put(row, 0, y);

                    if (row + 1 == minibatchSize) {

//                        if(minibatchSize==-1000){
//                            DoubleMatrix pca=klt_pca(X,50);
//                            System.out.println(pca);
//                        }

                        DoubleMatrix sm = X.mmul(theta);

                        if (pn + 1 + minibatchSize > shuffledPosts.size()) {//last
                            DoubleMatrix expY = MatrixFunctions.exp(Y).sub(1);
                            DoubleMatrix sub = MatrixFunctions.exp(sm).sub(1).sub(expY);
                            double v1 = variance(sub).sum();
                            double v2 = variance(expY).sum();
                            double r2 = v2 < 1e-6 ? 0 : (1 - v1 / v2) * 1000;
                            double cost = MatrixFunctions.pow(sub, 2).sum() / (minibatchSize);
                            if (parameters.regularization > 0) {

                                cost += parameters.regularization *
                                        MatrixFunctions.pow(theta.dup().put(1, 0.0), 2).sum() /
                                        (minibatchSize);
                            }

                            System.out.println(trainId + " epoch=" + epoch + " lr=" + learningRate + " batch=" + minibatchSize + " R2=" + r2 + " cost=" + cost);
//                            if(minibatchSize>90000 && minibatchSize<100000) {
//                                DoubleMatrix means = MatrixFunctions.pow(X.mulRowVector(theta).mul(minibatchSize/1000),2).columnMeans();
//                                System.err.println(means.toString());
//                            }

                        }
                        if (parameters.regularization > 0) {
                            theta = theta.sub(X.transpose().mmul(sm.sub(Y))
                                    .add(theta.dup().put(1, 0.0).mul(parameters.regularization))
                                    .mul(learningRate / minibatchSize));

                        } else {
                            theta = theta.sub(X.transpose().mmul(sm.sub(Y)).mul(learningRate / minibatchSize));
                        }

                    }

                }


            }

            if (epoch == 0) {
                int size = shuffledPosts.size();
                mean = sumF.div(size);
                spread = maxF.sub(minF);
                for (int j = 0; j < parameters.featuresLength; j++) {
                    double s = spread.get(j);
                    if (s < 1e-6) {
                        spread.put(j, 1);
                        mean.put(j, 0);
                    }
                }


            } else {
                if (Math.abs(learningRate * parameters.learningRateFactor) < 1) {
                    learningRate = learningRate * parameters.learningRateFactor;
                }
                if (parameters.minibatchFactor > 1.0) {
                    minibatchSize = (int) Math.ceil(minibatchSize * parameters.minibatchFactor);
                }
                if (minibatchSize >= maxbatchSize) minibatchSize = maxbatchSize;
                if (testPosts != null || groupPred != null) {
                    try {
                        LinearPredictor predictor = new LinearPredictor(theta, mean, spread, parameters);
                        if (groupPred != null) {
                            synchronized (groupPred) {
                                groupPred.put(groupPredKey, predictor);
                            }
                        }
                        if (testPosts != null) {
                            saveResults(predictor, posts, testPosts.values());
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }

        }
//        System.out.println(theta.toString());
        return new LinearPredictor(theta, mean, spread, parameters);
    }

    public static void fillFeatures(double[] f, Post post, Parameters parameters) {
//            double[] f=new double[len];
//            feats[pn]=f;
        Arrays.fill(f, 0);

//                if (pn % p == 0) System.out.println("features " + pn + "/" + ps);


//            double y = post.likes;
//            f[0] = post.likes;
        f[1] = 1;

        int i = 2;

        int unknownWords = 0;
        for (WordStatTf wordTf : post.words) {
            WordStat wordStat = wordTf.stat;
            if (wordStat != null) {
                int bowId = wordStat.bowId;
                if (bowId >= 0) {
                    double w;
                    if (parameters.tfidfBow) {
                        double tf = 0.5 + 0.5 * wordTf.tf / post.maxTf;
                        w = tf * wordStat.idf;
                    } else {
                        w = 1;
                    }
                    f[i + bowId] = w;
                } else {
                    unknownWords++;
                }
            }
        }

        i += parameters.bowDim;

        for (WordStatTf trigram : post.trigramsConsChars) {
            WordStat trigramStat = trigram.stat;
            if (trigramStat != null) {
                int bowTrigramId = trigramStat.bowId;
                if (bowTrigramId >= 0) {
                    double w;
                    if (parameters.tfidfBow) {
                        double tf = 0.5 + 0.5 * trigram.tf / post.maxTrigramTf;
                        w = tf * trigramStat.idf;
                    } else {
                        w = trigram.tf;
                    }
                    f[i + bowTrigramId] = w;
                }
            }
        }

        i += parameters.trigramDim;

        GroupStat groupStat = post.groupStat;
        if (groupStat != null && groupStat.bowid >= 0 && groupStat.bowid < parameters.groupDim) {
            f[i + groupStat.bowid] = 1;
        }
        i += parameters.groupDim;

        f[i + post.hour] = 1;
        i += 24;

        f[i + post.dayofweek] = 1;
        i += 7;

        f[i + post.month] = 1;
        i += 12;

        if (post.year >= 2005 && post.year <= 2014) {
            f[i + post.year - 2005] = 1;
        }

        i += 10;

        if (parameters.imagesWeights != null) {
            if (post.img != null) {
                for (long imgId : post.img) {
                    ImageWeight[] weights = parameters.imagesWeights.get(imgId);
                    if (weights != null) {
                        for (ImageWeight weight : weights) {
                            int classId = weight.imageClass.classId;
                            f[i + classId] += weight.imageWeight;
                        }
                    }
                }
            }
            i += parameters.imageDim;
        }

        if (parameters.sentence2vecDim > 0) {
            if (post.sentence2vec != null) {
                System.arraycopy(post.sentence2vec, 0, f, i, parameters.sentence2vecDim);
            }
            i += parameters.sentence2vecDim;
        }

        if (post.features != null) {

            System.arraycopy(post.features, 0, f, i, post.features.length);

        } else {

            int iFstart = i;

            f[i++] = post.txt.contains("!") ? 1 : 0;
            f[i++] = post.txt.contains("!!!") ? 1 : 0;
            f[i++] = post.txt.contains(":)") ? 1 : 0;
            f[i++] = post.txt.contains(":-)") ? 1 : 0;
            f[i++] = post.txt.contains(")") ? 1 : 0;
            f[i++] = post.txt.contains(")))") ? 1 : 0;

            f[i++] = post.img != null ? 1 : 0;
            f[i++] = post.txt.contains("Pool") ? 1 : 0;
            f[i++] = post.txt.contains("http") ? 1 : 0;
            f[i++] = post.txt.contains("youtube") ? 1 : 0;
            f[i++] = containsWord(post, "класс") ? 1 : 0;
            f[i++] = containsWord(post, "лайк") ? 1 : 0;


            int wordsn = post.wordsNum;
            if (wordsn < 1) wordsn = 1;
            int length = post.txt.length();
            if (length < 1) length = 1;
            double numvosk = getVosklic(post.txt);

            int is = i;

            f[i++] = post.capsWord;         //0
            f[i++] = (double) post.capsWord / wordsn;
            f[i++] = numvosk;
            f[i++] = numvosk / length;
            f[i++] = wordsn;

            f[i++] = length;                  // 5
            f[i++] = parameters.jan_01_2014 - post.time;
            f[i++] = post.img == null ? 0 : post.img.length;
            f[i++] = post.pool;
            f[i++] = (double) unknownWords / wordsn;
            f[i++] = (double) post.maxTf;


            if (groupStat != null) {
                f[i++] = groupStat.likesStat.likesum;  //11
                f[i++] = groupStat.likesStat.loglikesum;
                f[i++] = groupStat.likesStat.likesum / (groupStat.count + 1);
                f[i++] = groupStat.likesStat.loglikesum / (groupStat.count + 1);
            } else {
                i += 4;
            }
            f[i++] = post.frequencyForTenPostsInGroup;
            f[i++] = post.postsInGroupTomorrowDelta;
            f[i++] = post.postsInGroupHourDelta;

            f[i++] = 2014 - post.year;                 //18
            f[i++] = post.month;
            f[i++] = post.dayofmonth;
            f[i++] = post.dayofweek;
            f[i++] = post.hour;

            f[i++] = Math.sin(post.month * 2 * Math.PI / 12);   //23
            f[i++] = Math.sin((post.dayofmonth - 1) * 2 * Math.PI / 31);
            f[i++] = Math.sin(post.dayofweek * 2 * Math.PI / 7);
            f[i++] = Math.sin(post.hour * 2 * Math.PI / 24);

            f[i++] = Math.cos(post.month * 2 * Math.PI / 12);  //27
            f[i++] = Math.cos((post.dayofmonth - 1) * 2 * Math.PI / 31);
            f[i++] = Math.cos(post.dayofweek * 2 * Math.PI / 7);
            f[i++] = Math.cos(post.hour * 2 * Math.PI / 24);

            if (groupStat != null) {
                DateStat dayStat = getNearest(groupStat.dateStat, getDateStatKey(post.dayofmonth, post.month, post.year));
                DateStat monthStat = getNearest(groupStat.dateStat, getDateStatKey(0, post.month, post.year));

                f[i++] = dayStat.posts;  //31
                f[i++] = dayStat.likes;
                f[i++] = (double) dayStat.likes / (dayStat.posts + 1);
                f[i++] = monthStat.posts;
                f[i++] = monthStat.likes;
                f[i++] = (double) monthStat.likes / (monthStat.posts + 1);
            } else {
                i += 6;
            }

            DateStat dayStat = getNearest(parameters.totalDateStat, getDateStatKey(post.dayofmonth, post.month, post.year));
            DateStat monthStat = getNearest(parameters.totalDateStat, getDateStatKey(0, post.month, post.year));

            f[i++] = (double) dayStat.posts; //37
            f[i++] = (double) dayStat.likes;
            f[i++] = (double) dayStat.likes / (dayStat.posts + 1);
            f[i++] = (double) monthStat.posts;
            f[i++] = (double) monthStat.likes;
            f[i++] = (double) monthStat.likes / (monthStat.posts + 1);

            f[i++] = post.trigramsNum;
            f[i++] = post.maxTrigramTf;

            if (!parameters.noMapping) {
                for (int j = 0; j < i - is; j++) {    //43
                    double x = f[is + j];
                    f[i + j] = Math.log(Math.abs(x) + 1);
                    f[i + j + (i - is)] = x * x;
                    f[i + j + 2 * (i - is)] = x * x * x;
                }
            }

            post.features = new double[f.length - iFstart];
            System.arraycopy(f, iFstart, post.features, 0, post.features.length);


            for (int j = 0; j < f.length; j++) {
                double v = f[j];
                if (Double.isInfinite(v)) throw new RuntimeException("inf " + j);
                if (Double.isNaN(v)) throw new RuntimeException("Nan " + j);

            }


        }

    }

    private static DateStat getNearest(TreeMap<Integer, DateStat> dateStat, int key) {
        DateStat res1 = dateStat.get(key);
        if (res1 != null) {
            if (res1.posts > 1) {
                return res1;
            }
        }
        Map.Entry<Integer, DateStat> ceil = dateStat.ceilingEntry(key + 1);
        Map.Entry<Integer, DateStat> floor = dateStat.floorEntry(key - 1);
        if (ceil == null) {
            if (floor == null) {
                return res1;
            } else {
                return floor.getValue();
            }
        } else {
            if (floor == null) {
                return ceil.getValue();
            } else {
                if (Math.abs(key - ceil.getKey()) > Math.abs(key - floor.getKey())) {
                    return floor.getValue();
                } else {
                    return ceil.getValue();
                }
            }
        }


    }

    private static boolean containsWord(Post post, String word) {
        for (WordStatTf tf : post.words) {
            if (word.equals(tf.stat.word)) return true;
        }
        return false;
    }


    private static void computePostGroupTimeStat(Map<Integer, Post>... postss) {
        System.out.println("computePostGroupTimeStat");
        final Map<Integer, ArrayList<Post>> postsByGroupTime = new TreeMap<>();
        for (Map<Integer, Post> posts : postss) {
            for (Post post : posts.values()) {
                int groupid = post.groupid;
                ArrayList<Post> postsInGroup = postsByGroupTime.get(groupid);
                if (postsInGroup == null) {
                    postsInGroup = new ArrayList<>();
                    postsByGroupTime.put(groupid, postsInGroup);
                }
                postsInGroup.add(post);

            }

        }
        Comparator<Post> comparator = new Comparator<Post>() {
            @Override
            public int compare(Post o1, Post o2) {
                return Long.compare(o1.time, o2.time);
            }
        };
        int jj = 0;
        for (ArrayList<Post> posts : postsByGroupTime.values()) {
            Collections.sort(posts, comparator);
            int len = posts.size();
            for (int i = 0; i < len; i++) {
                if (++jj % 10000 == 0) System.out.println(jj);

                Post post = posts.get(i);
                int n = 0;
                int dayPosts = 0;
                int hourPosts = 0;
                long maxtime = post.time;
                final long tomorrow = post.time + 1000 * 3600 * 24;
                final long hourAfter = post.time + 1000 * 3600;
                int limit = 10;
                for (int j = i + 1; j < len; j++) {

                    Post afterPost = posts.get(j);
                    if (afterPost.groupid == post.groupid) {
                        if (n < limit) {
                            maxtime = afterPost.time;
                        }
                        if (afterPost.time <= tomorrow) {
                            dayPosts++;
                            if (afterPost.time <= hourAfter) {
                                hourPosts++;
                            }
                        }
                        n++;
                    }
                    if (n >= limit && afterPost.time > tomorrow) break;
                }
                post.postsInGroupTomorrowDelta = dayPosts;
                post.postsInGroupHourDelta = hourPosts;
                if (n >= limit) {
                    post.frequencyForTenPostsInGroup = 3600.0 * 1000.0 * 24.0 / (maxtime - post.time);
                }

            }
        }
    }

    private static List<String> getWordStrings(List<WordStat> words) {
        ArrayList<String> list = new ArrayList<String>(words.size());
        for (WordStat word : words) {
            list.add(word.word);
        }
        return list;
    }

    private static void printBestPosts(Map<Integer, Post> posts, int limit) {
        ArrayList<Post> bestposts = new ArrayList<Post>();
        for (Post post : posts.values()) {
            if (post.likes > limit) {
                bestposts.add(post);
            }
        }
        Collections.sort(bestposts, new Comparator<Post>() {
            @Override
            public int compare(Post o1, Post o2) {
                return Integer.compare(o1.likes, o2.likes);
            }
        });

        for (Post post : bestposts) {
            System.out.println(post.likes + "\t" + post.txt);
        }
    }

    private static int getVosklic(String txt) {
        int n = 0, l = txt.length();
        for (int i = 0; i < l; i++) {
            if (txt.charAt(i) == '!') n++;

        }
        return n;
    }

    private static int getCapsWord(List<String> words) {

        int n = 0;
        for (String word : words) {
            if (caps(word)) n++;
        }
        return n;
    }

    private static boolean caps(String word) {
        if (word.length() < 4) return false;
        char[] chars = word.toCharArray();
        for (char c : chars) {
            if (!Character.isUpperCase(c)) return false;
        }
        return true;
    }

    private static void parseWords(Tokeniser tokeniser,
                                   Map<String, WordStat> wordStatMap,
                                   Map<String, WordStat> consTrigramsMap,
                                   boolean bigrams,
                                   Map<Integer, Post>... postss) throws IOException {
        russianStemmer stemmer = new org.tartarus.snowball.ext.russianStemmer();
//        BufferedWriter w1 = new BufferedWriter(new FileWriter("/Volumes/Apple/imgok_posts.txt"));
//        BufferedWriter wStem = new BufferedWriter(new FileWriter("/Volumes/Apple/imgok_postsStem.txt"));
//        BufferedWriter wLower = new BufferedWriter(new FileWriter("/Volumes/Apple/imgok_postsLower.txt"));
        for (Map<Integer, Post> posts : postss) {


            int ps = posts.values().size();
            int p = ps / 100;
            if (p <= 0) p = 1;
            int pn = 0;
            for (Post post : posts.values()) {
                if (++pn % p == 0) System.out.println("word " + pn + "/" + ps);
                List<String> words = tokeniser.tokenize(post.txt);

                HashSet<String> wordsSet = new HashSet<>();
                List<String> wordsList = new ArrayList<String>();

                HashSet<String> trigramSet = new HashSet<>();
                List<String> trigramList = new ArrayList<String>();
                String prevword = null;
                for (String word : words) {
                    String wl = word.toLowerCase();
                    String stem = stem(stemmer, wl);
//                    w1.write(word + " ");
//                    wStem.write(stem + " ");
//                    wLower.write(wl + " ");

                    if (stem.length() < 2) {
                        prevword = null;
                        continue;
                    }
                    wordsSet.add(stem);
                    wordsList.add(stem);

                    if (bigrams && prevword != null) {
                        String bigram = prevword + " " + stem;
                        wordsSet.add(bigram);
                        wordsList.add(bigram);
                    }
                    prevword = stem;

                    String[] trigrams = getRusConsTrigrams(wl);
                    for (String trigram : trigrams) {
                        trigramSet.add(trigram);
                        trigramList.add(trigram);
                    }
                }

                TreeMap<String, WordStatTf> wordMap = new TreeMap<>();
                TreeMap<String, WordStatTf> trigramMap = new TreeMap<>();

                post.maxTf = getWordStatTfMap(wordMap, wordStatMap, post, wordsSet, wordsList);
                post.maxTrigramTf = getWordStatTfMap(trigramMap, consTrigramsMap, post, trigramSet, trigramList);

                Collection<WordStatTf> wordStatTfs = wordMap.values();
                Collection<WordStatTf> trigramsStatTfs = trigramMap.values();

                post.words = wordStatTfs.toArray(new WordStatTf[wordStatTfs.size()]);
                post.trigramsConsChars = trigramsStatTfs.toArray(new WordStatTf[trigramsStatTfs.size()]);
                post.trigramsNum = trigramList.size();
                post.wordsNum = words.size();
                post.capsWord = getCapsWord(words);

//                w1.write("\n");
//                wStem.write("\n");
//                wLower.write("\n");
            }


        }
//        w1.close();
//        wStem.close();
//        wLower.close();
    }

    private static int getWordStatTfMap(Map<String, WordStatTf> tfHashMapResult, Map<String, WordStat> wordStatHashMap, Post post, HashSet<String> wordsSet, List<String> wordsList) {
        int maxTf = 0;
        for (String w : wordsSet) {
            WordStat stat = wordStatHashMap.get(w);
            if (stat == null) {
                stat = new WordStat();
                stat.word = w;
                wordStatHashMap.put(w, stat);
            }
            WordStatTf tf = new WordStatTf();
            tf.stat = stat;
            tfHashMapResult.put(w, tf);
            stat.likesStat.addPost(post);
        }
        for (String w : wordsList) {
            WordStatTf tf = tfHashMapResult.get(w);
            tf.tf++;
            if (tf.tf > maxTf) maxTf = tf.tf;
        }
        return maxTf;
    }

    private static String stem(SnowballStemmer stemmer, String w) {
        stemmer.setCurrent(w.replace('ё', 'е'));
//        for (int i = 3; i != 0; i--) {
        boolean stem = stemmer.stem();
//            if(!stem)break;
//        }
        return stemmer.getCurrent();

    }

    private static Map<Integer, GroupStat> countGroups(Map<Integer, Post> posts, Map<Integer, DateStat> totalDateStat) {
        HashMap<Integer, GroupStat> groupStats = new HashMap<Integer, GroupStat>();

        for (Post post : posts.values()) {
            GroupStat groupStat = groupStats.get(post.groupid);
            if (groupStat == null) {
                groupStat = new GroupStat();
                groupStat.groupid = post.groupid;
                groupStats.put(post.groupid, groupStat);
            }
            groupStat.count++;
            groupStat.likesStat.addPost(post);
            post.groupStat = groupStat;
            addDateStat(false, post.dayofmonth, post.month, post.year, groupStat.dateStat);
            addDateStat(false, post.dayofmonth, post.month, post.year, totalDateStat);
        }
        return groupStats;
    }

    private static void addDateStat(boolean isLike, int day, int month, int year, Map<Integer, DateStat> dateStat) {
        int keyDay = getDateStatKey(day, month, year);
        int keyMonth = getDateStatKey(0, month, year);
        DateStat dayStat = dateStat.get(keyDay);
        if (dayStat == null) {
            dayStat = new DateStat();
            dateStat.put(keyDay, dayStat);
        }
        DateStat monthStat = dateStat.get(keyMonth);
        if (monthStat == null) {
            monthStat = new DateStat();
            dateStat.put(keyMonth, monthStat);
        }
        if (isLike) {
            dayStat.likes++;
            monthStat.likes++;
        } else {
            dayStat.posts++;
            monthStat.posts++;
        }


    }


    public static DoubleMatrix variance(DoubleMatrix input) {
        DoubleMatrix means = input.columnMeans();
        DoubleMatrix diff = MatrixFunctions.pow(input.subRowVector(means), 2);
//avg of the squared differences from the mean
        DoubleMatrix variance = diff.columnMeans().div(input.rows);
        return variance;

    }

    private static int getDateStatKey(int day, int month, int year) {
        int i = (year * 12 + month) * 32 + day;
        if (day == 0) i = -i;
        return i;
    }

    static double[] readCsvDoubleColumn(String fileName) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;
            ArrayList<Double> list = new ArrayList<>();
            while (null != (line = reader.readLine())) {
                try {
                    list.add(Double.valueOf(line.trim()));
                } catch (NumberFormatException e) {
                    e.printStackTrace();
                }
            }
            int len = list.size();
            double[] res = new double[len];
            for (int i = 0; i < res.length; i++) {
                res[i] = list.get(i);
            }
            return res;
        }
    }

    private static Map<Integer, Post> readPosts(String fileName) throws IOException {
        int postlimit = Integer.getInteger("ok.post.limit", -1);
        Map<Integer, Post> posts = new LinkedHashMap<>();
        CSVReader reader = new CSVReader(new FileReader(fileName), '\t', '\0');
        int line = 0;
        Post prevpost = null;
        System.out.println("Reading " + fileName + " ...");
        int j = 0;
        while (true) {
            if (++j % 10000 == 0) System.out.println(j);
            String[] next = reader.readNext();
            if (next == null) break;
            line++;
            if (postlimit > 0 && line > postlimit) break;
            if (next.length < 4) {
                if (next.length > 0) {
                    if (next.length < 2) {
                        prevpost.txt = prevpost.txt + " " + next[0];
                    }
                }
                continue;
            }
            Post post = new Post();
            post.id = Integer.valueOf(next[1]);
            post.groupid = Integer.valueOf(next[0]);
            post.time = Long.valueOf(next[2]);
            post.txt = String.valueOf(next[3]);
            prevpost = post;
            parseTime(post);
            parseImages(post);
            parsePool(post);

            post.vosklic = getVosklic(post.txt);

            posts.put(post.id, post);
        }
        reader.close();
        return posts;
    }

    private static void parseTime(HasTime hasTime) {
        GregorianCalendar calendar = new GregorianCalendar(TimeZone.getTimeZone("MSK"));
        calendar.setTimeInMillis(hasTime.time);
        hasTime.hour = calendar.get(Calendar.HOUR_OF_DAY);
        hasTime.dayofmonth = calendar.get(Calendar.DAY_OF_MONTH);
        hasTime.dayofweek = calendar.get(Calendar.DAY_OF_WEEK) - 1;
        hasTime.year = calendar.get(Calendar.YEAR);
        hasTime.month = calendar.get(Calendar.MONTH);

    }

    private static void readLikes(Map<Integer, Post> posts) throws IOException {
        CSVReader reader2 = new CSVReader(new FileReader("train_likes_count.csv"), ',');
        System.out.println("Reading likes count...");
        int j = 0;
        while (true) {
            if (++j % 10000 == 0) System.out.println(j);
            String[] next = reader2.readNext();
            if (next == null) break;
            if (next.length < 2) {
                continue;
            }
            try {
                int post_id = Integer.valueOf(next[0]);
                int likes = Integer.valueOf(next[1]);
                Post post = posts.get(post_id);
                if (post != null) {
                    post.likes = likes;
                    post.loglikes = Math.log(likes + 1);
                }
            } catch (NumberFormatException e) {
                //  e.printStackTrace();

            }
        }
        reader2.close();
    }

    private static void readLikes2(Map<Integer, Post> posts, Map<Integer, DateStat> totalDateStat) throws IOException {
        CSVReader reader2 = new CSVReader(new FileReader("train_likes.csv"), '\t');
        System.out.println("Reading likes time...");
        int j = 0;
        while (true) {
            if (++j % 1000000 == 0) System.out.println(j);
            String[] next = reader2.readNext();
            if (next == null) break;
            if (next.length < 3) {
                continue;
            }
            try {
                int user_id = Integer.valueOf(next[0]);
                int post_id = Integer.valueOf(next[1]);
                long time = Long.valueOf(next[2]);
                Post post = posts.get(post_id);
                if (post != null) {
                    HasTime likeTime = new HasTime();
                    likeTime.time = time;
                    parseTime(likeTime);
                    addDateStat(true, likeTime.dayofmonth, likeTime.month, likeTime.year, post.groupStat.dateStat);
                    addDateStat(true, likeTime.dayofmonth, likeTime.month, likeTime.year, totalDateStat);
                }
            } catch (NumberFormatException e) {
                e.printStackTrace();
            }
        }
        reader2.close();
    }

    private static void parsePool(Post post) {

        int i = 0;
        String st = "PoolAnswer[";
        while (true) {
            i = post.txt.indexOf(st, i + 1);
            if (i < 0) break;
            post.pool++;
        }

    }

    private static void parseImages(Post post) {

        String st = "Images[";
        int si = post.txt.indexOf(st);
        if (si >= 0) {
            si = si + st.length();
            int ei = post.txt.indexOf(']', si);
            if (ei > si) {
                String ls[] = post.txt.substring(si, ei).split(",");
                post.txt = post.txt.substring(0, si - st.length()) + post.txt.substring(ei + 1);
                ArrayList<Long> al = new ArrayList<Long>(ls.length);
                for (String s : ls) {
                    long l = 0;
                    try {
                        l = Long.valueOf(s.trim());
                    } catch (Exception e) {
                        e.printStackTrace();

                    }
                    if (l > 0 && !al.contains(l)) {
                        al.add(l);
                    }
                }
                if (al.size() > 0) {
                    post.img = new long[al.size()];
                    for (int i = 0; i < post.img.length; i++) {
                        post.img[i] = al.get(i);

                    }
                    Arrays.sort(post.img);
                }
            }
        }

    }

    static interface Tokeniser {
        List<String> tokenize(String txt);
    }

    public static class SpaceTokenizer implements Tokeniser {


        public List<String> tokenize(String txt) {
            return Arrays.asList(txt.split(" "));
        }
    }

    public static class TokenizerCharType implements Tokeniser {

        public List<String> tokenize(String text) {
            ArrayList<String> list = new ArrayList<String>();
            CharType ilp = CharType.WHITESPACE;
            int st = 0;
            char[] chars = text.toCharArray();
            for (int i = 0; i <= chars.length; i++) {
                CharType il;
                if (i < chars.length) {
                    char c = chars[i];
                    il = CharType.getType(c);
                    if (il == CharType.WHITESPACE) {
                        chars[i] = ' ';
                    }
                } else {
                    il = CharType.WHITESPACE;
                }
                if (il != ilp) {
                    int k = st;
                    int m = i;
                    while (k < m && chars[k] == ' ') k++;
                    while (k < m && chars[m - 1] == ' ') m--;
                    if (k < m) {
                        list.add(new String(chars, k, m - k));
                    }
                    st = i;
                }
                ilp = il;

            }
            return list;
        }


        private enum CharType {
            CYR,
            WHITESPACE,
            LAT,
            DIGIT,
            PUNCTUATION,
            PUNCTUATION2,
            VOSKL,
            UNKNOWN,
            HEART;

            private static CharType getType(char c) {
                if (c >= 'а' && c <= 'я') return CYR;
                if (c == ' ' || c == '\n') return WHITESPACE;
                if (c >= 'А' && c <= 'Я') return CYR;
                if (c >= 'a' && c <= 'z') return LAT;
                if (c >= 'A' && c <= 'Z') return LAT;
                if (c >= '0' && c <= '9') return DIGIT;
                if (".,".indexOf(c) >= 0) return PUNCTUATION;
                if (":-()".indexOf(c) >= 0) return PUNCTUATION2;
                if (c == '!') return VOSKL;
                if (c == 'ё') return CYR;
                if (c == 'Ё') return CYR;
                if (c == '♥') return HEART;
                return UNKNOWN;
            }

        }
    }

    public static DoubleMatrix klt_pca(DoubleMatrix x, int dim) {
        DoubleMatrix mean = x.columnMeans();
        DoubleMatrix b = x.subRowVector(mean);
        DoubleMatrix c = b.transpose().mmul(b).div(x.getColumns() - 1);
        ComplexDoubleMatrix[] eigenVectors = Eigen.eigenvectors(c);
        ComplexDoubleMatrix v = eigenVectors[0];
        ComplexDoubleMatrix d = eigenVectors[1];

        DoubleMatrix w = v.real().get(RangeUtils.all(), RangeUtils.interval(0, dim));
        DoubleMatrix s = MatrixFunctions.sqrt(c.diag());
        DoubleMatrix z = b.div(s);

        DoubleMatrix t = z.mmul(w);

        return t;
    }

    public static class ImageClass implements Serializable {
        int classId;
        String name;

        public String toString() {
            return name;
        }
    }

    public static class ImageWeight implements Serializable {
        ImageClass imageClass;
        double imageWeight;

        public String toString() {
            return imageClass.name + " : " + imageWeight;
        }
    }

    private static class ClassWeight {
        String className;
        double weight;

        public String toString() {
            return className + " : " + weight;
        }
    }

    public static Map<Long, ImageWeight[]> readImages(File dirImages, File allClasses, Parameters parameters) throws IOException {
        TreeMap<String, ImageClass> classes = new TreeMap<>();
        List<ClassWeight> weights = readWeight(allClasses);
        parameters.imgCls = new ImageClass[weights.size()];
        for (ClassWeight weight : weights) {
            ImageClass imageClass = new ImageClass();
            imageClass.name = weight.className;
            imageClass.classId = classes.size();
            parameters.imgCls[imageClass.classId] = imageClass;
            if (!classes.containsKey(imageClass.name)) {
                classes.put(imageClass.name, imageClass);
            }
        }
        parameters.imageDim = classes.size();
        Map<Long, ImageWeight[]> result;
        File imageDataFile = new File(System.getProperty("ok.imagesdata", "imagesdata.bin"));
        if (imageDataFile.exists()) {
            try (ObjectInputStream objectInStream = new ObjectInputStream(new FileInputStream(imageDataFile))) {
                result = (Map<Long, ImageWeight[]>) objectInStream.readObject();
                System.out.println("Read images from " + imageDataFile.getName());
                return result;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        result = new HashMap<>();
        File[] files = dirImages.listFiles();
        String postf = ".jpeg.ppm.txt";
        System.out.println("Reading images");
        int p = 0;
        for (File file : files) {
            if (++p % 10000 == 0) System.out.println(p + "/" + files.length);
            String fileName = file.getName();
            if (fileName.endsWith(postf)) {
                long imgId = Long.valueOf(fileName.substring(0, fileName.length() - postf.length()));
                weights = readWeight(file);
                ArrayList<ImageWeight> list = new ArrayList<ImageWeight>(weights.size());
                for (ClassWeight weight : weights) {
                    ImageClass imageClass = classes.get(weight.className);
                    if (imageClass != null) {
                        ImageWeight imageWeight = new ImageWeight();
                        imageWeight.imageClass = imageClass;
                        imageWeight.imageWeight = weight.weight;
                        list.add(imageWeight);
                    }
                }
                result.put(imgId, list.toArray(new ImageWeight[list.size()]));
            }
        }
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(imageDataFile))) {
            objectOutputStream.writeObject(result);
            System.out.println("Save images data to " + imageDataFile.getName());
        }
        return result;
    }

    private static List<ClassWeight> readWeight(File file) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            ArrayList<ClassWeight> weights = new ArrayList<ClassWeight>();
            while (null != (line = reader.readLine())) {
                int lastSpace = line.lastIndexOf(' ');
                if (lastSpace > 0) {
                    ClassWeight classWeight = new ClassWeight();
                    classWeight.className = line.substring(0, lastSpace);
                    classWeight.weight = Double.valueOf(line.substring(lastSpace + 1));
                    weights.add(classWeight);
                }
            }
            return weights;
        }
    }

    private static class PredictorGroupEnsemble implements Predictor {
        private final Map<Integer, Predictor> groupPred;

        public PredictorGroupEnsemble(Map<Integer, Predictor> groupPred) {
            this.groupPred = new TreeMap<Integer, Predictor>(groupPred);
        }

        @Override
        public double predict(Post post) {
            int g = post.groupid;
            Predictor predictor = groupPred.get(g);
            if (predictor == null) {
                predictor = groupPred.get(0);
            }
            return predictor.predict(post);
        }
    }

    private static String[] getRusConsTrigrams(String wordLowerCase) {
        String cons = "нпрстклмбвгдшжзйфхчц";
        char[] ch = wordLowerCase.toCharArray();
        char[] conss = new char[ch.length];
        int cj = 0;
        for (int j = 0; j < ch.length; j++) {
            char c = ch[j];
            if (c == 'щ') c = 'ш';
            if (cons.indexOf(c) >= 0) {
                conss[cj++] = c;
            }
        }

        if (cj >= 3) {
            String[] res = new String[cj - 2];
            for (int i = 0; i < res.length; i++) {
                res[i] = new String(conss, i, 3);
            }
            return res;
        } else return new String[0];
    }


    private static List<Post> findClosestSent2vec(Post post, int limit, Map<Integer, Post>... posts) {
        HashMap<Integer, Double> mapsDistance = new HashMap<>();
        double[] sentence2vec = post.sentence2vec;
        double norm = 0;
        for (int i = 0; i < sentence2vec.length; i++) {
            norm += sentence2vec[i] * sentence2vec[i];
        }
        norm = Math.sqrt(norm);
        List<Post> sort = new ArrayList<>();
        for (Map<Integer, Post> postMap : posts) {
            for (Map.Entry<Integer, Post> e : postMap.entrySet()) {
                Post p = e.getValue();
                double cos = 0;
                double[] sentence2vec2 = p.sentence2vec;
                for (int i = 0; i < sentence2vec.length; i++) {
                    cos += sentence2vec2[i] * sentence2vec[i];
                }
                double norm2 = 0;
                for (int i = 0; i < sentence2vec2.length; i++) {
                    norm2 += sentence2vec2[i] * sentence2vec2[i];
                }
                norm2 = Math.sqrt(norm2);
                cos = cos / (norm * norm2);


                mapsDistance.put(p.id, cos);
            }
            sort.addAll(postMap.values());
        }

        Collections.sort(sort, new Comparator<Post>() {
            @Override
            public int compare(Post o1, Post o2) {
                return Double.compare(mapsDistance.get(o2.id), mapsDistance.get(o1.id));
            }
        });

        if (sort.size() > limit) {
            return new ArrayList<>(sort.subList(0, limit));
        } else {
            return sort;
        }

    }

}
