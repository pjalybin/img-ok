package ok;

import java.util.TreeMap;

/**
* @author Petr Zhalybin
* @since 31.08.2014 13:35
*/
class GroupStat {
    int groupid;
    int count;
    LikesStat likesStat = new LikesStat();
    int bowid = -1;
    TreeMap<Integer, DateStat> dateStat = new TreeMap<Integer, DateStat>();
}
