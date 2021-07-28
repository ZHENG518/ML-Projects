def selectionSort(nums):
    """
    选择排序
    - 依次选择第N小的数和N位置上的数交换位置
    - 时间复杂度 N2 空间复杂度 1
    """
    for i in range(len(nums)):
        min_ind = i
        for j in range(i+1, len(nums)):
            if nums[j]<nums[min_ind]:
                min_ind = j
        nums[i], nums[min_ind] = nums[min_ind], nums[i]

def bubbleSort(nums):
    """
    冒泡排序
    - 重复的从头开始遍历数组，两两比较相邻的元素，将大的元素向右交换
    - 时间复杂度 N2 空间复杂度 1
    """
    for i in range(len(nums), 0, -1):
        for j in range(0, i-1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]

def insertionSort(nums):
    """
    插入排序
    - 从第二个元素开始,向左一一比较，直到找到正确的位置
    """
    for i in range(1, len(nums)):
        cur_num = nums[i]
        j = i-1
        while j>=0 and cur_num<nums[j]:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = cur_num

def mergeSort(nums):
    """
    归并排序
    - 递归的将列表分为两部分，将两部分排序，再合并起来
    - 时间复杂度 NlogN 空间复杂度 N
    """
    def merge(left,right):
        result = []
        while left and right:
            if left[0] <= right[0]:
                result.append(left.pop(0))
            else:
                result.append(right.pop(0))
        while left:
            result.append(left.pop(0))
        while right:
            result.append(right.pop(0))
        return result

    if len(nums)<2: return nums # 递归结束条件
    middle = len(nums)//2
    left, right = nums[:middle], nums[middle:]
    return merge(mergeSort(left), mergeSort(right))

def quickSort(nums):

    def sort(nums, begin, end):
        if begin >= end:
            return nums
        pivot = nums[begin]
        left = begin
        right = end
        while left < right:
            while left<right and nums[right] >= pivot:
                right -= 1
            nums[left] = nums[right]
            while left<right and nums[left] < pivot:
                left += 1
            nums[right] = nums[left]
        nums[right] = pivot
        sort(nums, begin, right-1)
        sort(nums, right+1, end)
        return nums

    return sort(nums, 0, len(nums)-1)







if __name__ == '__main__':
    nums = [64, 25, 12, 22, 11, 22]
    nums = quickSort(nums)
    print(nums)