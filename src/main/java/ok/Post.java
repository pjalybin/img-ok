package ok;

/**
* @author Petr Zhalybin
* @since 31.08.2014 13:34
*/
class Post extends HasTime {
    String txt;
    long[] img;
    int id;
    int groupid;
    int likes;
    double loglikes;
    int pool;
    WordStatTf[] words;
    public int capsWord;
    public int vosklic;
    GroupStat groupStat;


    double frequencyForTenPostsInGroup;
    double postsInGroupTomorrowDelta;
    double postsInGroupHourDelta;

    double[] features;
    public int wordsNum;
    public int maxTf;
    public WordStatTf[] trigramsConsChars;
    public int trigramsNum;
    public int maxTrigramTf;

    double[] sentence2vec;

    public String toString() {
        return txt;
    }
}
