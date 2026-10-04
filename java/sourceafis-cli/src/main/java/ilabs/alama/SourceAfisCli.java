package ilabs.alama;

import com.machinezoo.sourceafis.FingerprintCompatibility;
import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Usage:
 *   java -jar sourceafis-cli.jar match-iso   probe.iso gallery.iso
 *   java -jar sourceafis-cli.jar match-image probe.png gallery.png [dpi]
 *
 * Prints the SourceAFIS similarity score on stdout. match-iso compares templates made
 * by the alama pipeline; match-image lets SourceAFIS run its own extractor on images.
 */
public final class SourceAfisCli {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: match-iso <probe> <gallery> | match-image <probe> <gallery> [dpi]");
            System.exit(2);
        }
        FingerprintTemplate probe;
        FingerprintTemplate gallery;
        switch (args[0]) {
            case "match-iso" -> {
                probe = FingerprintCompatibility.importTemplate(Files.readAllBytes(Path.of(args[1])));
                gallery = FingerprintCompatibility.importTemplate(Files.readAllBytes(Path.of(args[2])));
            }
            case "match-image" -> {
                double dpi = args.length > 3 ? Double.parseDouble(args[3]) : 500;
                probe = fromImage(Path.of(args[1]), dpi);
                gallery = fromImage(Path.of(args[2]), dpi);
            }
            default -> {
                System.err.println("unknown command: " + args[0]);
                System.exit(2);
                return;
            }
        }
        System.out.println(new FingerprintMatcher(probe).match(gallery));
    }

    private static FingerprintTemplate fromImage(Path path, double dpi) throws Exception {
        return new FingerprintTemplate(
            new FingerprintImage(Files.readAllBytes(path), new FingerprintImageOptions().dpi(dpi)));
    }
}
