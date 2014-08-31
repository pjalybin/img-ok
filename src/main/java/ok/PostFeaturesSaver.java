package ok;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collection;

/**
* @author Petr Zhalybin
* @since 31.08.2014 13:36
*/
enum PostFeaturesSaver {
    ARFF {
        void saveFeatures(Collection<Post> posts, Parameters parameters, File file) throws IOException {
            int n = 0;
            try (PrintWriter filewriter = new PrintWriter(file)) {
                filewriter.println("@relation img-ok-" + file.getName());
                String[] fn = App.getFeatureNames(parameters);
                for (int i = 0; i < fn.length; i++) {
                    if (parameters.skipColumns != null && parameters.skipColumns[i]) continue;
                    filewriter.println("@attribute " + fn[i].replace(' ', '_') + " numeric");
                }
                filewriter.println("@attribute id numeric");
                filewriter.println("@attribute groupid numeric");
                filewriter.println("@attribute likes numeric");
                filewriter.println("@attribute floor" + ((int) parameters.logScale) + "loglikes numeric");
                filewriter.println("@data");
                double[] f = new double[parameters.featuresLength + 4];
                for (Post post : posts) {
                    if (n++ % 20000 == 0) System.out.println("save features arff " + file.getName() + " " + n);
                    App.fillFeatures(f, post, parameters);
                    f[parameters.featuresLength] = post.id;
                    f[parameters.featuresLength + 1] = post.groupid;
                    f[parameters.featuresLength + 2] = post.likes;
                    f[parameters.featuresLength + 3] = Math.floor(post.loglikes * parameters.logScale);
                    StringBuilder sb = new StringBuilder();
                    sb.append('{');
                    for (int i = 0; i < f.length; i++) {
                        if (parameters.skipColumns != null && parameters.skipColumns[i]) continue;
                        double v = f[i];
                        if (Double.isInfinite(v) || Double.isNaN(v))
                            throw new RuntimeException("Infinite " + post.id + " " + i + " " + fn[i]);
                        if (v != 0.0) {
                            if (sb.length() > 1) {
                                sb.append(',');
                            }
                            sb.append(i).append(' ');
                            if (Math.rint(v) == v && Math.abs(v)<2e9) {
                                sb.append((int) v);
                            } else {
                                sb.append(v);
                            }
                        }
                    }
                    sb.append('}');
                    filewriter.println(sb.toString());
                }
            }
        }
    },
    VW {
        void saveFeatures(Collection<Post> posts, Parameters parameters, File file) throws IOException {
            int n = 0;
            try (PrintWriter filewriter = new PrintWriter(file)) {

                double[] f = new double[parameters.featuresLength];
                for (Post post : posts) {
                    if (n++ % 20000 == 0) System.out.println("save features VW " + file.getName() + " " + n);
                    App.fillFeatures(f, post, parameters);
//                        int logClass = (int)(Math.floor(post.loglikes * parameters.logScale)+1);
                    StringBuilder sb = new StringBuilder();
                    sb.append(post.likes).append(' ');
                    sb.append('\'').append(post.id).append(' ');
                    sb.append('|');
                    for (int i = 2; i < f.length; i++) {
                        if(parameters.skipColumns!=null && parameters.skipColumns[i])continue;
                        double v = f[i];
                        if(Double.isInfinite(v) || Double.isNaN(v))throw new RuntimeException("Infinite "+post.id+" "+i);
                        if (v != 0.0) {
                            sb.append(' ');
                            sb.append(i);
                            if(v!=1.0) {
                                sb.append(':');
                                if (Math.rint(v) == v && Math.abs(v)<2e9) {
                                    sb.append((int) v);
                                } else {
                                    sb.append(v);
                                }
                            }
                        }
                    }
                    filewriter.println(sb.toString());
                }
            }
        }
    },
    CSV {
        void saveFeatures(Collection<Post> posts, Parameters parameters, File file) throws IOException {
            int n = 0;
            try (PrintWriter filewriter = new PrintWriter(file)) {
                StringBuilder sb = new StringBuilder();
                sb.append("\"id\",\"groupid\",\"likes\",\"floor" + ((int) parameters.logScale) + "loglikes\"");
                double[] f = new double[parameters.featuresLength];
                String[] fn = App.getFeatureNames(parameters);
                for (int i = 1; i < fn.length; i++) {
                    if (parameters.skipColumns != null && parameters.skipColumns[i]) continue;
                    sb.append(",\"").append(fn[i].replace(' ', '_').replace('\n', '_').replace('"', '_')).append('"');
                }
                filewriter.println(sb.toString());
                for (Post post : posts) {
                    if (n++ % 20000 == 0) System.out.println("save features " + file.getName() + " " + n);
                    App.fillFeatures(f, post, parameters);
                    f[0] = Math.floor(post.loglikes * parameters.logScale);
                    sb = new StringBuilder();
                    sb.append(post.id);
                    sb.append(',').append(post.groupid);
                    sb.append(',').append(post.likes);
                    for (int i = 0; i < f.length; i++) {
                        if (parameters.skipColumns != null && parameters.skipColumns[i] && i > 0) continue;
                        double v = f[i];
                        sb.append(',');
                        if (v == 0.0) {
                            sb.append('0');
                        } else {
                            if (Double.isInfinite(v) || Double.isNaN(v))
                                throw new RuntimeException("Infinite " + post.id + " " + i + " " + fn[i]);
                            double fv = Math.rint(v);
                            if (fv == v && Math.abs(v)<2e9) {
                                sb.append((int) fv);
                            } else {
                                sb.append(v);
                            }
                        }
                    }
                    filewriter.println(sb.toString());
                }
            }
        }
    };

    abstract void saveFeatures(Collection<Post> posts, Parameters parameters, File file) throws IOException;
}
