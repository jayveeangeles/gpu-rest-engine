#include "helper.h"

std::vector<int> nms(std::vector<std::pair<float, int> >& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold)
{
	auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
		if (x1min > x2min) {
			std::swap(x1min, x2min);
			std::swap(x1max, x2max);
		}
		return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
	};
	auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
		float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
		float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
		float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
		float overlap2D = overlapX * overlapY;
		float u = area1 + area2 - overlap2D;
		return u == 0 ? 0 : overlap2D / u;
	};

	std::vector<int> indices;
	for (auto i : score_index)
	{
		const int idx = i.second;
		bool keep = true;
		for (unsigned k = 0; k < indices.size(); ++k)
		{
			if (keep)
			{
				const int kept_idx = indices[k];
				float overlap = computeIoU(&bbox[(idx*numClasses + classNum) * 4],
					&bbox[(kept_idx*numClasses + classNum) * 4]);
				keep = overlap <= nms_threshold;
			}
			else
				break;
		}
		if (keep) indices.push_back(idx);
	}
	return indices;
}

void bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo,
	const int N, const int nmsMaxOut, const int numCls)
{
	float width, height, ctr_x, ctr_y;
	float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	float *deltas_offset, *predBBoxes_offset, *imInfo_offset;
	for (int i = 0; i < N * nmsMaxOut; ++i)
	{
		width = rois[i * 4 + 2] - rois[i * 4] + 1;
		height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
		ctr_x = rois[i * 4] + 0.5f * width;
		ctr_y = rois[i * 4 + 1] + 0.5f * height;
		deltas_offset = deltas + i * numCls * 4;
		predBBoxes_offset = predBBoxes + i * numCls * 4;
		imInfo_offset = imInfo + i / nmsMaxOut * 3;
		for (int j = 0; j < numCls; ++j)
		{
			dx = deltas_offset[j * 4];
			dy = deltas_offset[j * 4 + 1];
			dw = deltas_offset[j * 4 + 2];
			dh = deltas_offset[j * 4 + 3];
			pred_ctr_x = dx * width + ctr_x;
			pred_ctr_y = dy * height + ctr_y;
			pred_w = exp(dw) * width;
			pred_h = exp(dh) * height;
			predBBoxes_offset[j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
		}
	}
}

void getBbox(std::vector<box*>& boxes, float **outputData, float nmsThresh, float detectThresh, float* imInfo, int batchSize, const int nmsMaxOut, const int outputClsSize)
{
	const int outputBBoxSize = outputClsSize * 4;
	float* bboxPreds = outputData[0];
	float* clsProbs = outputData[1]; 
	float *rois = outputData[2]; 
	// predicted bounding boxes
	float* predBBoxes = new float[batchSize * nmsMaxOut * outputBBoxSize];

	// unscale back to raw image space
	for (int i = 0; i < batchSize; ++i)
	{
		float * rois_offset = rois + i * nmsMaxOut * 4;
		for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
			rois_offset[j] /= imInfo[i * 3 + 2];
	}

	bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, batchSize, nmsMaxOut, outputClsSize);

	for (int i = 0; i < batchSize; ++i)
	{
		float *bbox = predBBoxes + i * nmsMaxOut * outputBBoxSize;
		float *scores = clsProbs + i * nmsMaxOut * outputClsSize;
		for (int c = 1; c < outputClsSize; ++c) // skip the background
		{
			std::vector<std::pair<float, int> > score_index;
			for (int r = 0; r < nmsMaxOut; ++r)
			{
				if (scores[r*outputClsSize + c] > detectThresh)
				{
					score_index.push_back(std::make_pair(scores[r*outputClsSize + c], r));
					std::stable_sort(score_index.begin(), score_index.end(),
						[](const std::pair<float, int>& pair1,
							const std::pair<float, int>& pair2) {
						return pair1.first > pair2.first;
					});
				}
			}

			// apply NMS algorithm
			std::vector<int> indices = nms(score_index, bbox, c, outputClsSize, nmsThresh);
			box *tmp;
			// Show results
			for (unsigned k = 0; k < indices.size(); ++k)
			{
				int idx = indices[k];
				float x1 = bbox[idx*outputBBoxSize + c * 4];
				float y1 = bbox[idx*outputBBoxSize + c * 4 + 1];
				float x2 = bbox[idx*outputBBoxSize + c * 4 + 2];
				float y2 = bbox[idx*outputBBoxSize + c * 4 + 3];
				
				tmp = new box(x1,x2,y1,y2,c,scores[idx*outputClsSize + c]);
				boxes.push_back(tmp);
			}
		}
	}
}