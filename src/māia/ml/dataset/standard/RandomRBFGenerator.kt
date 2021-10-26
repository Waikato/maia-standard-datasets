package māia.ml.dataset.standard

import māia.ml.dataset.DataMetadata
import māia.ml.dataset.DataRow
import māia.ml.dataset.DataStream
import māia.ml.dataset.headers.DataColumnHeaders
import māia.ml.dataset.headers.MutableDataColumnHeaders
import māia.ml.dataset.headers.ensureOwnership
import māia.ml.dataset.type.DataRepresentation
import māia.ml.dataset.type.standard.Nominal
import māia.ml.dataset.type.standard.Numeric
import māia.util.assertType
import māia.util.inlineRangeForLoop
import māia.util.magnitude
import māia.util.nextDoubleArray
import māia.util.nextGaussian
import māia.util.nextIntWeighted
import kotlin.random.Random

/**
 * TODO: What class does.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
class RandomRBFGenerator(
    modelSeed: Int = 1,
    instanceSeed: Int = 1,
    private val numClasses: Int = 2,
    private val numAttributes: Int = 10,
    numCentroids: Int = 50
) : DataStream<DataRow> {

    private val centroids: Array<Centroid>

    private val centroidWeights: DoubleArray

    private val instanceRandom : Random = Random(instanceSeed)

    override val metadata : DataMetadata = object : DataMetadata {
        override val name : String
            get() = this@RandomRBFGenerator.toString()
    }

    override val numColumns : Int = numAttributes + 1

    init {
        val modelRand = Random(modelSeed)
        centroids = Array(numCentroids) {
            Centroid(
                modelRand.nextDoubleArray(numAttributes),
                modelRand.nextInt(numClasses),
                modelRand.nextDouble()
            )
        }
        centroidWeights = modelRand.nextDoubleArray(numCentroids)
    }

    private val headersInternal = MutableDataColumnHeaders(numColumns).also { headers ->
        inlineRangeForLoop(numAttributes) { index ->
            headers.append("att ${index + 1}", Numeric.PlaceHolder(false), false)
        }
        headers.append("class", Nominal.PlaceHolder(false, *Array(numClasses) { "class ${it + 1}" }), true)
    }

    override val headers = headersInternal.readOnlyView

    override fun rowIterator() : Iterator<DataRow> {
        return object : Iterator<DataRow> {
            override fun hasNext() : Boolean = true
            override fun next() : DataRow = this@RandomRBFGenerator.nextRow()
        }
    }

    private fun nextRow(): DataRow {
        val centroid = centroids[instanceRandom.nextIntWeighted(centroidWeights)]
        val attVals = DoubleArray(numAttributes) { instanceRandom.nextDouble(-1.0, 1.0) }
        val desiredMag = instanceRandom.nextGaussian() * centroid.stdDev
        val scale = desiredMag / attVals.magnitude
        for (index in 0 until numAttributes)
            attVals[index] = attVals[index] * scale + centroid.centre[index]

        return DataRow(headers, attVals, centroid.classIndex)
    }

    class DataRow(
        override val headers: DataColumnHeaders,
        private val attributeValues: DoubleArray,
        private val classIndex: Int
    ): māia.ml.dataset.DataRow {
        override fun <T> getValue(
            representation : DataRepresentation<*, *, out T>
        ) : T = headers.ensureOwnership(representation) {
            columnIndex.let {
                if (it < attributeValues.size)
                    convert(attributeValues[it], assertType<Numeric<*, *>>(dataType).canonicalRepresentation)
                else
                    convert(classIndex, assertType<Nominal<*, *, *, *>>(dataType).indexRepresentation)
            }
        }
    }
}

private data class Centroid(
    val centre: DoubleArray,
    val classIndex: Int,
    val stdDev: Double
)
