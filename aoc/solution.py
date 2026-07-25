class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []

        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            max_element = max(subarray)

            if all(subarray[j] < subarray[j+1] for j in range(k-1)) and all(subarray[j] == subarray[j+1] - 1 for j in range(k-1)):
                results.append(max_element)
            else:
                results.append(-1)

        return results
