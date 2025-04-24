// Example 1: Zero-Cost Abstractions with Iterators vs Manual Loops
// In many languages, using higher-order functions comes with a performance penalty
// In Rust, the iterator version often compiles to the same efficient machine code as manual loops

// Manual loop approach (what you might do in other languages for performance)
fn sum_even_squares_manual(numbers: &[i32]) -> i32 {
    let mut sum = 0;
    for i in 0..numbers.len() {
        let num = numbers[i];
        if num % 2 == 0 {
            sum += num * num;
        }
    }
    sum
}

// Iterator approach (more readable, yet just as fast in Rust)
fn sum_even_squares_iterator(numbers: &[i32]) -> i32 {
    numbers
        .iter()
        .filter(|&&x| x % 2 == 0)
        .map(|&x| x * x)
        .sum()
}

// Example 2: Memory Management Without Garbage Collection
// Demonstrating how Rust's ownership system eliminates common performance issues

// In many languages: Hidden allocations, copies, and GC pressure
fn process_text_other_lang(text: String) -> String {
    // Implicitly creates multiple copies and temporary allocations
    let lines = text.split('\n'); // Hidden allocation in many languages
    let filtered = lines.filter(|line| !line.trim().is_empty()); // Another allocation
    let processed = filtered.map(|line| line.trim().to_uppercase()); // More allocations
    processed.collect::<Vec<_>>().join("\n") // Final allocation
}

// In Rust: Explicit and efficient with zero hidden costs
fn process_text_rust(text: &str) -> String {
    text.split('\n')
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_uppercase())
        .collect::<Vec<_>>()
        .join("\n")
}

// Example 3: String Handling - Slices vs Owned Data
// Demonstrating how Rust's type system guides you toward efficient choices

// Inefficient approach - unnecessary ownership transfers and allocations
fn extract_domain_inefficient(email: String) -> String {
    let parts: Vec<&str> = email.split('@').collect();
    if parts.len() != 2 {
        return String::new();
    }
    parts[1].to_string()
}

// Efficient approach - using references to avoid allocations
fn extract_domain_efficient(email: &str) -> &str {
    if let Some(at_pos) = email.find('@') {
        &email[at_pos + 1..]
    } else {
        ""
    }
}

// Example 4: Compile-time Evaluation
// Rust's const generics and const functions enable computation at compile time

// Computing Fibonacci at runtime (like most languages would)
fn fibonacci_runtime(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }

    let mut a = 0;
    let mut b = 1;
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

// Computing Fibonacci at compile time
const fn fibonacci_compile_time(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        n => {
            let mut a = 0;
            let mut b = 1;
            let mut i = 2;
            while i <= n {
                let temp = a + b;
                a = b;
                b = temp;
                i += 1;
            }
            b
        }
    }
}

// Using compile-time evaluation
const FIB_10: u64 = fibonacci_compile_time(10); // Computed at compile time!

// Example 5: Safe Concurrency Without Performance Overhead
// Demonstrating how Rust's ownership rules enable efficient parallelism

use std::thread;

// In many languages: Locks, synchronization, or copying data
fn parallel_sum_other_langs(data: &[i32]) -> i32 {
    // Would typically involve locks, mutexes, or copying data
    // Often leads to contention or excessive memory usage
    // Implementation omitted for brevity
    0 // Placeholder
}

// In Rust: Efficient parallelism with compile-time safety
fn parallel_sum_rust(data: &[i32]) -> i32 {
    if data.len() < 1000 {
        // For small arrays, just do it sequentially
        return data.iter().sum();
    }

    // For larger arrays, split the work
    let mid = data.len() / 2;
    let (left, right) = data.split_at(mid);

    // Process each half in parallel with zero-copy slices
    let left_handle = thread::spawn(move || left.iter().sum::<i32>());

    let right_sum = right.iter().sum::<i32>();
    let left_sum = left_handle.join().unwrap();

    left_sum + right_sum
}

// Example 6: Avoiding Dynamic Dispatch with Generics and Traits
// Showing how Rust can eliminate runtime type checking overhead

// In many languages: Runtime polymorphism with performance cost
trait ShapeOtherLang {
    fn area(&self) -> f64;
}

struct CircleOtherLang {
    radius: f64,
}

impl ShapeOtherLang for CircleOtherLang {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

// Would incur virtual dispatch overhead in most languages
fn total_area_dynamic(shapes: &[Box<dyn ShapeOtherLang>]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// In Rust: Static dispatch with generics - zero overhead
trait Shape {
    fn area(&self) -> f64;
}

struct Circle {
    radius: f64,
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

// No runtime dispatch needed - monomorphization creates optimized code for each type
fn total_area<T: Shape>(shapes: &[T]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// Example 7: Move Semantics - Avoiding Hidden Copies
// Demonstrating how Rust's move semantics prevent unexpected performance issues

struct LargeData {
    // Imagine this contains megabytes of data
    payload: Vec<u8>,
}

// In many languages: Implicit, expensive copying
fn process_data_other_lang(data: LargeData) -> LargeData {
    // Many languages would implicitly copy 'data' here
    // This could be very expensive but hidden from the developer
    LargeData {
        payload: data.payload, // More copying!
    }
}

// In Rust: Explicit transfers of ownership
fn process_data_rust(mut data: LargeData) -> LargeData {
    // Ownership is moved, no hidden copies
    // We can modify in place for efficiency
    data.payload.push(42);
    data // Transfer ownership back to caller
}

// Or, even more efficient when appropriate
fn process_data_rust_ref(data: &mut LargeData) {
    // No ownership transfer at all, just borrowed access
    data.payload.push(42);
}

// Example 8: Inlining and Optimizing Small Functions
// Demonstrating how Rust's compiler aggressively optimizes code

// This small function will be inlined automatically
#[inline]
fn add_one(x: i32) -> i32 {
    x + 1
}

fn process_numbers(numbers: &[i32]) -> Vec<i32> {
    // The compiler will inline add_one, eliminating function call overhead
    numbers.iter().map(|&x| add_one(x)).collect()
}

// Example 9: Stack vs Heap Allocation
// Showing how Rust encourages stack allocation for better performance

// In many languages: Hidden heap allocations
fn create_array_other_lang() -> Vec<i32> {
    // Most languages would heap-allocate this array
    vec![1, 2, 3, 4, 5]
}

// In Rust: Explicit choice between stack and heap
fn create_array_stack() -> [i32; 5] {
    // Stack allocated - no allocation overhead, better cache locality
    [1, 2, 3, 4, 5]
}

fn create_array_heap() -> Vec<i32> {
    // Explicitly heap allocated when needed
    vec![1, 2, 3, 4, 5]
}

// Example 10: Compile-time Bounds Checking Elimination
// Showing how Rust's compiler can eliminate unnecessary bounds checks

fn sum_array(arr: &[i32]) -> i32 {
    let mut sum = 0;

    // The compiler can prove this loop is safe and eliminate bounds checks
    for i in 0..arr.len() {
        sum += arr[i]; // No runtime bounds check needed!
    }

    sum
}

// More advanced: Using unsafe when appropriate with clear safety boundaries
fn sum_array_optimized(arr: &[i32]) -> i32 {
    let mut sum = 0;

    // Explicitly avoiding bounds checks when we're certain it's safe
    unsafe {
        for i in 0..arr.len() {
            sum += *arr.get_unchecked(i);
        }
    }

    sum
}
