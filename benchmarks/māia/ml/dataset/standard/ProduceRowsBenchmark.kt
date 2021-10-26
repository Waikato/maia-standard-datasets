package māia.ml.dataset.standard

import kotlinx.benchmark.Blackhole
import kotlinx.benchmark.Mode
import kotlinx.benchmark.Scope
import māia.ml.dataset.type.standard.Numeric
import māia.util.assertType
import māia.util.inlineRangeForLoop
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Measurement
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.annotations.Warmup
import java.util.concurrent.TimeUnit

@Warmup(iterations = 0)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@BenchmarkMode(Mode.AverageTime)
open class RandomDisposeBenchmark {

    @State(Scope.Benchmark)
    open class Params {

        @Param("100", "10000")
        var numRows: Int = 0

        @Param("10",  "100")
        var numAttrs: Int = 0

        @Param("10",  "100")
        var numClasses: Int = 0

        @Param("50",  "150")
        var numCentroids: Int = 0
    }

    @Benchmark
    fun maiaBenchmark(params: Params, blackhole : Blackhole) {
        val gen = RandomRBFGenerator(
            numClasses = params.numClasses,
            numAttributes = params.numAttrs,
            numCentroids = params.numCentroids
        )
        val iter = gen.rowIterator()

        inlineRangeForLoop(params.numRows) {
            val row = iter.next()
            var sum = 0.0
            inlineRangeForLoop(params.numAttrs) {
                val repr = assertType<Numeric<*, *>>(gen.headers[it].type).canonicalRepresentation
                sum += row.getValue(repr)
            }
            blackhole.consume(sum)
        }
    }

}
